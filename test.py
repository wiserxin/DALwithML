from neural.util import Trainer, Loader
from neural.models import CNN
import torch
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from neural.util.evaluator import Evaluator
import random


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

def theLoss(x, target):
    # print('x:',x.size())
    # print('target:', target.size())
    # return nn.BCELoss()(x.squeeze(2), target)
    # return nn.BCELoss()(x, target)
    return nn.MultiLabelSoftMarginLoss()(x, target)

eurlex_path = r"../../datasets/eurLex/"
rcv2_path = r"../../datasets/rcv2/"

Loader = Loader()
data = Loader.load_rcv2( datapath=rcv2_path, vocab_size=30000)

train_data = data['train_points']
val_data = data['test_points']

train_data = train_data[:8000]
# too small the valdata amount so ...
val_data = val_data[:2000]


model = CNN(  word_vocab_size = 30000
            , word_embedding_dim = 300
            , word_out_channels = 200
            , output_size = 103
            , pretrained = data['embed']
            )
model = nn.DataParallel(model).cuda()

trainer = Trainer(model,
                  r'./result/test/',
                  "CNN",
                  top_k=40,
                  eval_begin= 1
                  )

test_performance = trainer.train_supervisedLearning(60,
                                                    train_data,
                                                    val_data,
                                                    0.004,
                                                    checkpoint_path=r'./result/test/',
                                                    batch_size=2048,
                                                    )

print(trainer.eval_value)