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


# CNN-Kim
class CNN_KIM(nn.Module):
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
        # 得到的是概率值，如果要预测值，需要sigmoid后，经过阈值筛选得到 (0,1)值

    def predict(self, x, usecuda=True):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv13)
        x2 = self.conv_and_pool(x, self.conv14)
        x3 = self.conv_and_pool(x, self.conv15)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        output = self.linear(x)
        output = torch.sigmoid(output) > 0.5
        return output

    def features(self,x,usecuda=True):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv13)
        x2 = self.conv_and_pool(x, self.conv14)
        x3 = self.conv_and_pool(x, self.conv15)
        x = torch.cat((x1, x2, x3), 1)
        return x

    def features_with_pred(self,x,usecuda=True):
        x = self.features(x,usecuda)
        output = self.linear(x)
        return torch.cat((x,output), 1)

# XML-CNN
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
        self.conv13 = nn.Conv2d(1, word_out_channels, (3, word_embedding_dim), stride=2)
        self.conv14 = nn.Conv2d(1, word_out_channels, (4, word_embedding_dim), stride=2)
        self.conv15 = nn.Conv2d(1, word_out_channels, (5, word_embedding_dim), stride=2)
        self.pool13 = nn.MaxPool1d(self.out_size(word_embedding_dim, 3, stride=2)//4, stride=16)
        self.pool14 = nn.MaxPool1d(self.out_size(word_embedding_dim, 4, stride=2)//4, stride=16)
        self.pool15 = nn.MaxPool1d(self.out_size(word_embedding_dim, 5, stride=2)//4, stride=16)

        self.dropout = nn.Dropout(p=dropout_p)

        hidden_size = 4600
        self.linear1 = nn.Linear(hidden_size, 512)
        self.linear2 = nn.Linear(512, output_size)

    def out_size(self, l_in, kernel_size, padding=0, dilation=1, stride=1):
        a = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
        b = int(a / stride)
        return b + 1

    def conv_and_relu(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        return x

    def forward(self, x, usecuda=True):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_relu(x,self.conv13)
        x2 = self.conv_and_relu(x, self.conv14)
        x3 = self.conv_and_relu(x, self.conv15)
        # print("{} size: {}".format("x", x.size()))
        # print("{} size: {}".format("x1", x1.size()))
        # print("{} size: {}".format("x2", x2.size()))
        # print("{} size: {}".format("x3", x3.size()))
        # x size: torch.Size([400, 1, 300, 300])
        # x1 size: torch.Size([400, 200, 149])
        # x2 size: torch.Size([400, 200, 149])
        # x3 size: torch.Size([400, 200, 148])
        x1 = self.pool13(x1)
        x2 = self.pool14(x2)
        x3 = self.pool15(x3)
        # print("{} size: {}".format("x", x.size()))
        # print("{} size: {}".format("x1", x1.size()))
        # print("{} size: {}".format("x2", x2.size()))
        # print("{} size: {}".format("x3", x3.size()))
        # x size: torch.Size([400, 1, 300, 300])
        # x1 size: torch.Size([400, 200, 8])
        # x2 size: torch.Size([400, 200, 8])
        # x3 size: torch.Size([400, 200, 7])
        x = torch.cat((x1, x2, x3), 2)
        x = x.view(x.size()[0],-1 )
        x = self.dropout(x)

        hidden = self.linear1(x)
        output = self.linear2(hidden)
        # print("{} size: {}".format("x", x.size()))
        # print("{} size: {}".format("hidden", hidden.size()))
        # print("{} size: {}".format("output", output.size()))
        return output

    def predict(self, x, usecuda=True):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_relu(x, self.conv13)
        x2 = self.conv_and_relu(x, self.conv14)
        x3 = self.conv_and_relu(x, self.conv15)
        x = torch.cat((x1, x2, x3), 2)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        hidden = self.linear1(x)
        output = self.linear2(hidden)
        output = torch.sigmoid(output) > 0.5
        return output

    def features(self,x,usecuda=True):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_relu(x, self.conv13)
        x2 = self.conv_and_relu(x, self.conv14)
        x3 = self.conv_and_relu(x, self.conv15)
        x = torch.cat((x1, x2, x3), 2)
        x = x.view(x.size()[0], -1)
        return x


class xml_cnn2(nn.Module):
    # 严格符合定义的
    # 需要增加 dropout 层 且重新组织输入的params
    def __init__(self, params, embedding_weights):
        super(xml_cnn, self).__init__()

        self.params = params

        stride = params["stride"]
        emb_dim = embedding_weights.shape[1]
        hidden_dims = params["hidden_dims"]
        sequence_length = params["sequence_length"]
        filter_channels = params["filter_channels"]
        d_max_pool_p = params["d_max_pool_p"]

        self.filter_sizes = params["filter_sizes"]

        # 層の定義
        self.lookup = nn.Embedding.from_pretrained(embedding_weights, freeze=False)

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        fin_l_out_size = 0

        # self.dropout_0 = nn.Dropout(p=0.25)
        # self.dropout_1 = nn.Dropout(p=0.5)
        self.batch_norm_0 = nn.BatchNorm1d(emb_dim)

        for fsz, n, ssz in zip(self.filter_sizes, d_max_pool_p, stride):
            conv_n = nn.Conv1d(emb_dim, filter_channels, fsz, stride=ssz)
            torch.nn.init.kaiming_normal_(conv_n.weight)

            conv_out_size = self.out_size(sequence_length, fsz, stride=ssz)
            pool_k_size = conv_out_size // n
            assert conv_out_size / n != 0

            # Dynamic Max-Pooling
            pool_n = nn.MaxPool1d(pool_k_size, stride=pool_k_size)

            pool_out_size = filter_channels * n
            fin_l_out_size += pool_out_size

            self.conv_layers.append(conv_n)
            self.pool_layers.append(pool_n)

        self.l1 = nn.Linear(fin_l_out_size, params["hidden_dims"])
        self.batch_norm_2 = nn.BatchNorm1d(hidden_dims)
        self.l2 = nn.Linear(hidden_dims, params["num_of_class"])

        # Heの初期値でWeightsを初期化
        torch.nn.init.kaiming_normal_(self.l1.weight)
        torch.nn.init.kaiming_normal_(self.l2.weight)

    # 畳み込み後のTensorのサイズを計算
    def out_size(self, l_in, kernel_size, padding=0, dilation=1, stride=1):
        a = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
        b = int(a / stride)
        return b + 1

    def forward(self, x):
        # Embedding層
        h_non_static = self.lookup.forward(x.permute(1, 0))
        # h_non_static = self.dropout_0(h_non_static)
        h_non_static = h_non_static.permute(0, 2, 1)

        h_non_static = self.batch_norm_0(h_non_static)

        h_list = []

        # Conv, Pooling層
        for i in range(len(self.filter_sizes)):
            h_n = self.conv_layers[i](h_non_static)
            h_n = h_n.view(h_n.shape[0], 1, h_n.shape[1] * h_n.shape[2])
            h_n = self.pool_layers[i](h_n)
            h_n = F.relu(h_n)
            h_n = h_n.view(h_n.shape[0], -1)
            h_list.append(h_n)
            del h_n

        if len(self.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]

        # Full Connected層
        h = F.relu(self.l1(h))

        # h = self.dropout_1(h)
        h = self.batch_norm_2(h)

        # Output層
        y = self.l2(h)
        return y