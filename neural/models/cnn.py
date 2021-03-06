import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

import neural
from neural.util import Initializer
from neural.util import Loader
from neural.modules import EncoderCNN
from neural.modules import EncoderCNN_Pair

class CNN_ori(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_out_channels, output_size, 
                 dropout_p = 0.5, pretrained=None, double_embedding = False, cuda_device=0):
        
        super(CNN, self).__init__()
        self.cuda_device = cuda_device
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels
        
        self.initializer = Initializer()
        # self.loader = Loader()

        self.embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))

        #CNN
        self.docs_encoder = EncoderCNN(word_vocab_size, word_embedding_dim, word_out_channels)


        self.dropout = nn.Dropout(p=dropout_p)
        
        hidden_size = word_out_channels
        self.linear = nn.Linear(hidden_size, output_size)
        
        #self.lossfunc = nn.CrossEntropyLoss()


    def forward(self, docs, tags=None, usecuda=True):

        # 2019-4-6
        docs_embedded = self.embedding(docs)

        docs_features = self.docs_encoder(docs_embedded)


        features = self.dropout(docs_features)
        output = self.linear(features)
        #loss = self.lossfunc(output, tags)
        
        return output#loss


    # def predict(self, docs, scoreonly=False, usecuda=True, encoder_only = False):
    #
    #     docs_embedded = self.embedding(docs)
    #
    #     docs_features = self.docs_encoder(docs_embedded)
    #
    #     #2019-5-11
    #     if encoder_only:
    #         return docs_features.data.cpu().numpy(),
    #
    #     features = self.dropout(docs_features)
    #     output = self.linear(features)
    #
    #     scores = torch.max(F.sigmoid(output, dim =1), dim=1)[0].data.cpu().numpy()
    #     if scoreonly:
    #         return scores
    #
    #     prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
    #     return scores, prediction

    def predict_score(self, docs):

        docs_embedded = self.embedding(docs)
        docs_features = self.docs_encoder(docs_embedded)

        features = self.dropout(docs_features)
        output = self.linear(features)
        return F.softmax(output, dim=1)


class CNN(nn.Module):

    def __init__(self, word_vocab_size, word_embedding_dim, word_out_channels, output_size,
                 dropout_p=0.5, pretrained=None, double_embedding=False, cuda_device=0):
        super(CNN, self).__init__()
        self.cuda_device = cuda_device
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels

        self.initializer = Initializer()
        # self.loader = Loader()

        self.embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))

        # CNN
        self.conv13 = nn.Conv2d(1, word_out_channels, (3, word_embedding_dim))
        self.conv14 = nn.Conv2d(1, word_out_channels, (4, word_embedding_dim))
        self.conv15 = nn.Conv2d(1, word_out_channels, (5, word_embedding_dim))

        self.dropout = nn.Dropout(p=dropout_p)

        hidden_size = word_out_channels*3
        self.linear = nn.Linear(hidden_size, output_size)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, usecuda=True):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_pool(x,self.conv13)
        x2 = self.conv_and_pool(x,self.conv14)
        x3 = self.conv_and_pool(x,self.conv15)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        output = self.linear(x)
        return output

