from neural.util.loader import Loader
from neural.models import CNN
import torch
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from neural.util.evaluator import Evaluator


def theLoss(x, target):
    # print('x:',x.size())
    # print('target:', target.size())
    # return nn.BCELoss()(x.squeeze(2), target)
    # return nn.BCELoss()(x, target)
    return nn.MultiLabelSoftMarginLoss()(x, target)

eurlex_path = r"../../datasets/eurLex/"

Loader = Loader()
data = Loader.load_eurlex( datapath=eurlex_path, vocab_size=30000)

X_trn, Y_trn, Y_trn_o = data['train']
X_tst, Y_tst, Y_tst_o = data['test']


num_classes = Y_trn.shape[1]


model = CNN(  word_vocab_size = 30000
            , word_embedding_dim = 300
            , word_out_channels = 200
            , output_size = num_classes
            , pretrained = data['embed']
            )
model = nn.DataParallel(model).cuda()

optimizer = Adam(model.parameters(), lr=0.001)

evaluator = Evaluator("CNN")
best_eval = 0.0
now_eval  = 0.0

tr_batch_size = 256

for epoch in range(20):
    torch.cuda.empty_cache()

    nr_trn_num = X_trn.shape[0]
    nr_batches = int(np.ceil(nr_trn_num / float(tr_batch_size)))


    model.train()
    model_loss = 0
    for iteration, batch_idx in enumerate(np.random.permutation(range(nr_batches))):

        start_idx = batch_idx * tr_batch_size
        end_idx = min((batch_idx + 1) * tr_batch_size, nr_trn_num)

        X = X_trn[start_idx:end_idx]
        X = Variable(torch.from_numpy(X).long()).cuda()
        Y = Y_trn[start_idx:end_idx]

        batch_target = Variable(torch.from_numpy(Y.A.astype(int)).float()).cuda()
        optimizer.zero_grad()
        output = model(X)
        loss = theLoss(output, batch_target)
        loss.backward()
        model_loss = loss.item()
        optimizer.step()
        torch.cuda.empty_cache()


        print("\rEpoch: {} Iteration: {}/{} ({:.1f}%)  Loss: {:.5f} {:.5f}".format(
                      epoch,iteration, nr_batches,
                      iteration * 100 / nr_batches,
                      model_loss, 0),
                      end="")

    if  not (epoch%5):
        print('')
        best_eval, now_eval, save = evaluator.evaluate(model, data['test'], best_eval, )
        print("\rEpoch: {} Loss:{:.5f} Best_NDCG5:{:.5f} NDCG5:{:.5f}\n".format(epoch, model_loss, best_eval, now_eval))
    torch.cuda.empty_cache()

