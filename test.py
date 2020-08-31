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
aapd_path = r"../../datasets/aapd/"

Loader = Loader()
# data = Loader.load_rcv2( datapath=rcv2_path, vocab_size=30000)
data = Loader.load_aapd( datapath=aapd_path, vocab_size=30000)
train_data = data['train_points']
val_data = data['test_points']
train_data = train_data[:30000]
val_data = val_data[:]



model = CNN(  word_vocab_size = 30000
            , word_embedding_dim = 300
            , word_out_channels = 200
            , output_size = 54
            , pretrained = data['embed']
            )
model = nn.DataParallel(model).cuda()

trainer = Trainer(model,
                  r'./result',
                  "CNN",
                  top_k=40,
                  eval_begin= 1
                  )

test_performance = trainer.train_supervisedLearning(40,
                                                    train_data,
                                                    val_data,
                                                    0.004,
                                                    checkpoint_path=r'./result',
                                                    batch_size=512,
                                                    )


#########################################  load stack 测试 #######################################

# load stack data
# stack_path = r"../../datasets/stackOverflow/"
# data,label2id = Loader.load_stack(stack_path,)
# print(len(data),len(label2id))
# count = 0
# for key in label2id.keys():
#     if label2id[key][1] > 1000:
#         print("\"{}\":{}".format(key,count),end=",")
#         count += 1
# print(count)

# load aapd data 测试
# aapd_path = r"../../datasets/aapd/"
# docs,label2id,a = Loader.load_aapd(aapd_path)
# print(a)
# print(len(label2id))
# print(label2id)