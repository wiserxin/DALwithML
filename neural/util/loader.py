from __future__ import print_function
import os
import torch

import copy

import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from .utils import *
from .w2v import load_word2vec
import codecs
import pickle
import itertools
import scipy.sparse as sp
import string

def removePunctuation(text):
    temp = []
    for c in text:
        if c not in string.punctuation:
            temp.append(c)
    newText = ''.join(temp)
    return newText

def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = removePunctuation(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data, label2id):
    x_text = [doc['text'] for doc in data]
    x_text = [s.split(" ") for s in x_text]
    labels = [doc['catgy'] for doc in data]
    row_idx, col_idx, val_idx = [], [], []
    for i in range(len(labels)):
        l_list = list(set(labels[i]))  # remove duplicate cateories to avoid double count
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    # m = max(row_idx) + 1
    # n = max(col_idx) + 1
    m = len(data)
    n = len(label2id)
    Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    print('Y shape : {} * {}'.format(m, n))
    return [x_text, Y, labels]


def pad_sentences(sentences, padding_word="<PAD/>", max_length=300):
    # sequence_length = min(max(len(x) for x in sentences), max_length)
    sequence_length = max_length
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < max_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, vocab_size=50000):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # append <UNK/> symbol to the vocabulary
    vocabulary['<UNK/>'] = len(vocabulary)
    vocabulary_inv.append('<UNK/>')
    return [vocabulary, vocabulary_inv, word_counts]


def build_input_data(sentences, vocabulary):
    x = np.array([
        [vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in
        sentences])
    # x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
    return x


class Loader(object):

    def __init__(self):
        pass

    def word_mapping(self, dataset):

        p = [k[0] + ' ' + k[1] for k in dataset]
        words = [[x.lower() for x in s.split()] for s in p]

        # Create word list
        dico = create_dico(words)

        dico['<PAD>'] = 10000001
        dico['<UNK>'] = 10000000

        # Keep only words that appear more than 1
        dico = {k: v for k, v in dico.items() if v >= 2}

        word_to_id, id_to_word = create_mapping(dico)

        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in words)
        ))
        return dico, word_to_id, id_to_word

    def load_eurlex(self, datapath, sents_max_len = 500, vocab_size = 50000):

        # 读取已缓存数据
        if os.path.exists(os.path.join(datapath, 'eurlexLoaded.pkl')):
            with open(os.path.join(datapath, 'eurlexLoaded.pkl'), 'rb') as f:
                return pickle.load(f)

        def clean_str(string):
            # remove stopwords
            # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip().lower()

        def load_data_and_labels(data):
            x_text = [clean_str(doc['text']) for doc in data]
            x_text = [s.split(" ") for s in x_text]
            labels = [doc['catgy'] for doc in data]
            row_idx, col_idx, val_idx = [], [], []
            for i in range(len(labels)):
                l_list = list(set(labels[i]))  # remove duplicate cateories to avoid double count
                for y in l_list:
                    row_idx.append(i)
                    col_idx.append(y)
                    val_idx.append(1)
            # m = max(row_idx) + 1
            # n = max(col_idx) + 1
            m = len(data)
            n = 3954 # eurlex数据集一共有3954个label
            Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
            print("Y shape:",Y.shape)
            return [x_text, Y, labels]

        def pad_sentences(sentences, padding_word="<PAD/>", max_length=500):
            sequence_length = min(max(len(x) for x in sentences), max_length)
            padded_sentences = []
            for i in range(len(sentences)):
                sentence = sentences[i]
                if len(sentence) < max_length:
                    num_padding = sequence_length - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                else:
                    new_sentence = sentence[:max_length]
                padded_sentences.append(new_sentence)
            return padded_sentences

        def build_vocab(sentences, vocab_size=50000):
            word_counts = Counter(itertools.chain(*sentences))
            vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
            vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
            # append <UNK/> symbol to the vocabulary
            vocabulary['<UNK/>'] = len(vocabulary)
            vocabulary_inv.append('<UNK/>')
            return [vocabulary, vocabulary_inv, word_counts]

        def build_input_data(sentences, vocabulary):
            x = np.array([
                [vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in
                 sentences])
            # x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
            return x

        path = os.path.join(datapath, 'eurlex_raw_text.p')
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            [train, test, _vocab, _catgy] = pickle.load( f )

        trn_sents, Y_trn, Y_trn_o = load_data_and_labels(train)
        tst_sents, Y_tst, Y_tst_o = load_data_and_labels(test)

        trn_sents_padded = pad_sentences(trn_sents, max_length=sents_max_len)
        tst_sents_padded = pad_sentences(tst_sents, max_length=sents_max_len)

        vocabulary, vocabulary_inv, vocabulary_count = build_vocab(trn_sents_padded + tst_sents_padded, vocab_size=vocab_size)

        X_trn = build_input_data(trn_sents_padded, vocabulary)
        X_tst = build_input_data(tst_sents_padded, vocabulary)

        r = {'train': (X_trn, Y_trn, Y_trn_o),
                'test' : (X_tst, Y_tst, Y_tst_o),
               'vocab' : (vocabulary, vocabulary_inv, vocabulary_count),
                'embed': load_word2vec(datapath ,'glove', vocabulary_inv, 300),
                'train_points': [(X_trn[i], Y_trn[i], Y_trn_o[i], i)  for i in range(len(Y_trn_o))],
                'test_points' : [(X_tst[i], Y_tst[i], Y_tst_o[i], i)  for i in range(len(Y_tst_o))]
                }

        with open(os.path.join(datapath, 'eurlexLoaded.pkl'), 'wb') as f:
            pickle.dump(r,f)

        return r


    def load_rcv2(self,datapath,  sents_max_len = 300, vocab_size = 50000):

        # 读取已缓存数据
        if os.path.exists( os.path.join(datapath,'rcv2Loaded.pkl') ):
            with open(os.path.join(datapath,'rcv2Loaded.pkl') , 'rb') as f:
                return pickle.load(f)


        def clean_str(string):
            # remove stopwords
            # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip().lower()

        def load_data_and_labels(data,label2id):
            x_text = [doc['text'] for doc in data]
            x_text = [s.split(" ") for s in x_text]
            labels = [doc['catgy'] for doc in data]
            row_idx, col_idx, val_idx = [], [], []
            for i in range(len(labels)):
                l_list = list(set(labels[i]))  # remove duplicate cateories to avoid double count
                for y in l_list:
                    row_idx.append(i)
                    col_idx.append(y)
                    val_idx.append(1)
            # m = max(row_idx) + 1
            # n = max(col_idx) + 1
            m = len(data)
            n = len(label2id)
            Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
            print('Y shape : {} * {}'.format(m,n))
            return [x_text, Y, labels]

        def pad_sentences(sentences, padding_word="<PAD/>", max_length=300):
            # sequence_length = min(max(len(x) for x in sentences), max_length)
            sequence_length = max_length
            padded_sentences = []
            for i in range(len(sentences)):
                sentence = sentences[i]
                if len(sentence) < max_length:
                    num_padding = sequence_length - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                else:
                    new_sentence = sentence[:max_length]
                padded_sentences.append(new_sentence)
            return padded_sentences

        def build_vocab(sentences, vocab_size=50000):
            word_counts = Counter(itertools.chain(*sentences))
            vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
            vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
            # append <UNK/> symbol to the vocabulary
            vocabulary['<UNK/>'] = len(vocabulary)
            vocabulary_inv.append('<UNK/>')
            return [vocabulary, vocabulary_inv, word_counts]

        def build_input_data(sentences, vocabulary):
            x = np.array([
                [vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in
                 sentences])
            # x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
            return x

        # [{'text':"...", 'catgy':['cat1','cat2',...] },]
        file_names = ['train_texts.txt','train_labels.txt','test_texts.txt','test_labels.txt']
        label2id = {}

        counter = 0
        train = []
        path1 = os.path.join(datapath, file_names[0])
        path2 = os.path.join(datapath, file_names[1])
        assert os.path.exists(path1)
        assert os.path.exists(path2)
        with open(path1, 'r') as f1:
            with open(path2,'r') as f2:
                while True:
                    text = f1.readline()[:-1]
                    labels = f2.readline().split()
                    if text=='' and labels==[]:
                        break
                    else:
                        # print(counter)
                        counter += 1
                        for i in labels:
                            if i not in label2id.keys():
                                label2id[i] = len(label2id)
                        train.append({'text':text, 'catgy':[ label2id[i] for i in labels ]})


        test = []
        path1 = os.path.join(datapath, file_names[2])
        path2 = os.path.join(datapath, file_names[3])
        assert os.path.exists(path1)
        assert os.path.exists(path2)
        with open(path1, 'r') as f1:
            with open(path2,'r') as f2:
                while True:
                    text = f1.readline()[:-1]
                    labels = f2.readline().split()
                    if text=='' and labels==[]:
                        break
                    else:
                        for i in labels:
                            if i not in label2id.keys():
                                label2id[i] = len(label2id)
                        test.append({'text':text, 'catgy':[ label2id[i] for i in labels ]})


        # build catgy map from      str to id
        # and transform it

        trn_sents, Y_trn, Y_trn_o = load_data_and_labels(train,label2id)
        tst_sents, Y_tst, Y_tst_o = load_data_and_labels(test,label2id)

        # trn_sents_len = [len(i) for i in trn_sents]
        # tst_sents_len = [len(i) for i in tst_sents]
        # with open('rcv2_featrues.txt','w') as f:
        #     for i in trn_sents_len:
        #         print(i,file=f)
        #     print('..........',file=f)
        #     for i in tst_sents_len:
        #         print(i,file=f)

        trn_sents_padded = pad_sentences(trn_sents, max_length=sents_max_len)
        tst_sents_padded = pad_sentences(tst_sents, max_length=sents_max_len)

        vocabulary, vocabulary_inv, vocabulary_count = build_vocab(trn_sents_padded + tst_sents_padded, vocab_size=vocab_size)

        X_trn = build_input_data(trn_sents_padded, vocabulary)
        X_tst = build_input_data(tst_sents_padded, vocabulary)

        r =  {'train': (X_trn, Y_trn, Y_trn_o),
                'test' : (X_tst, Y_tst, Y_tst_o),
               'vocab' : (vocabulary, vocabulary_inv, vocabulary_count),
                'embed': load_word2vec(datapath ,'glove', vocabulary_inv, 300),
                'train_points': [(X_trn[i], Y_trn[i], Y_trn_o[i], i)  for i in range(len(Y_trn_o))],
                'test_points' : [(X_tst[i], Y_tst[i], Y_tst_o[i], i)  for i in range(len(Y_tst_o))]
                }
        with open(os.path.join(datapath, 'rcv2Loaded.pkl'), 'wb') as f:
            pickle.dump(r,f)
        return r


    def load_rcv1(self,datapath,  sents_max_len = 300, vocab_size = 50000):
        # trn 775220
        # tst 1191


        # 读取已缓存数据
        if os.path.exists( os.path.join(datapath,'rcv1Loaded.pkl') ):
            with open(os.path.join(datapath,'rcv1Loaded.pkl') , 'rb') as f:
                return pickle.load(f)


        def clean_str(string):
            # remove stopwords
            # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip().lower()

        def load_data_and_labels(data,label2id):
            x_text = [doc['text'] for doc in data]
            x_text = [s.split(" ") for s in x_text]
            labels = [doc['catgy'] for doc in data]
            row_idx, col_idx, val_idx = [], [], []
            for i in range(len(labels)):
                l_list = list(set(labels[i]))  # remove duplicate cateories to avoid double count
                for y in l_list:
                    row_idx.append(i)
                    col_idx.append(y)
                    val_idx.append(1)
            # m = max(row_idx) + 1
            # n = max(col_idx) + 1
            m = len(data)
            n = len(label2id)
            Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
            print('Y shape : {} * {}'.format(m,n))
            return [x_text, Y, labels]

        def pad_sentences(sentences, padding_word="<PAD/>", max_length=300):
            # sequence_length = min(max(len(x) for x in sentences), max_length)
            sequence_length = max_length
            padded_sentences = []
            for i in range(len(sentences)):
                sentence = sentences[i]
                if len(sentence) < max_length:
                    num_padding = sequence_length - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                else:
                    new_sentence = sentence[:max_length]
                padded_sentences.append(new_sentence)
            return padded_sentences

        def build_vocab(sentences, vocab_size=50000):
            word_counts = Counter(itertools.chain(*sentences))
            vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
            vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
            # append <UNK/> symbol to the vocabulary
            vocabulary['<UNK/>'] = len(vocabulary)
            vocabulary_inv.append('<UNK/>')
            return [vocabulary, vocabulary_inv, word_counts]

        def build_input_data(sentences, vocabulary):
            x = np.array([
                [vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in
                 sentences])
            # x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
            return x

        # [{'text':"...", 'catgy':['cat1','cat2',...] },]
        file_names = ['train.src','train.tgt','test.src','test.tgt']
        label2id = {}

        counter = 0
        train = []
        path1 = os.path.join(datapath, file_names[0])
        path2 = os.path.join(datapath, file_names[1])
        assert os.path.exists(path1)
        assert os.path.exists(path2)
        with open(path1, 'r') as f1:
            with open(path2,'r') as f2:
                while True:
                    text = f1.readline()[:-1]
                    labels = f2.readline().split()
                    if text=='' and labels==[]:
                        break
                    else:
                        # print(counter)
                        counter += 1
                        for i in labels:
                            if i not in label2id.keys():
                                label2id[i] = len(label2id)
                        train.append({'text':text, 'catgy':[ label2id[i] for i in labels ]})


        test = []
        path1 = os.path.join(datapath, file_names[2])
        path2 = os.path.join(datapath, file_names[3])
        assert os.path.exists(path1)
        assert os.path.exists(path2)
        with open(path1, 'r') as f1:
            with open(path2,'r') as f2:
                while True:
                    text = f1.readline()[:-1]
                    labels = f2.readline().split()
                    if text=='' and labels==[]:
                        break
                    else:
                        for i in labels:
                            if i not in label2id.keys():
                                label2id[i] = len(label2id)
                        test.append({'text':text, 'catgy':[ label2id[i] for i in labels ]})


        # build catgy map from      str to id
        # and transform it

        trn_sents, Y_trn, Y_trn_o = load_data_and_labels(train,label2id)
        tst_sents, Y_tst, Y_tst_o = load_data_and_labels(test,label2id)

        # trn_sents_len = [len(i) for i in trn_sents]
        # tst_sents_len = [len(i) for i in tst_sents]
        # with open('rcv2_featrues.txt','w') as f:
        #     for i in trn_sents_len:
        #         print(i,file=f)
        #     print('..........',file=f)
        #     for i in tst_sents_len:
        #         print(i,file=f)

        trn_sents_padded = pad_sentences(trn_sents, max_length=sents_max_len)
        tst_sents_padded = pad_sentences(tst_sents, max_length=sents_max_len)

        vocabulary, vocabulary_inv, vocabulary_count = build_vocab(trn_sents_padded + tst_sents_padded, vocab_size=vocab_size)

        X_trn = build_input_data(trn_sents_padded, vocabulary)
        X_tst = build_input_data(tst_sents_padded, vocabulary)

        r =  {'train': (X_trn, Y_trn, Y_trn_o),
                'test' : (X_tst, Y_tst, Y_tst_o),
               'vocab' : (vocabulary, vocabulary_inv, vocabulary_count),
                'embed': load_word2vec(datapath ,'glove', vocabulary_inv, 300),
                'train_points': [(X_trn[i], Y_trn[i], Y_trn_o[i], i)  for i in range(len(Y_trn_o))],
                'test_points' : [(X_tst[i], Y_tst[i], Y_tst_o[i], i)  for i in range(len(Y_tst_o))]
                }
        with open(os.path.join(datapath, 'rcv1Loaded.pkl'), 'wb') as f:
            pickle.dump(r,f)
        return r


    def load_stack(self,datapath,  sents_max_len = 300, vocab_size = 50000):
        # stack oveerflow data


        # 读取已缓存数据
        if os.path.exists( os.path.join(datapath,'stackLoaded.pkl') ):
            with open(os.path.join(datapath,'stackLoaded.pkl') , 'rb') as f:
                return pickle.load(f)

        # [{'text':"...", 'catgy':['cat1','cat2',...] },]
        import csv
        file_names = ['stackdata_utf8.csv']
        label2id = {}

        counter = 0
        train = []
        path = os.path.join(datapath, file_names[0])
        assert os.path.exists(path)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                data.append(row)

        # 使用全局所有数据
        label2id = dict() # {name: [id,count]}
        temp = []
        # 第一行是表头 #row:  id	question	answer 	tags
        # for i in range(1,len(data)):
        #     text = data[i][1] + ' ' + data[i][2]
        #     text = clean_str(text)
        #
        #     tags = data[i][3]
        #     # 为了处理c#,需要倒转再倒转
        #     tags = tags[::-1].split("###")[:-1]
        #     if len(tags) == 1:
        #         continue # 跳过只有一个标签的数据
        #     tags = [ onetag[::-1] for onetag in tags ]
        #     for onetag in tags:
        #         if onetag not in label2id.keys():
        #             label2id[onetag] = [len(label2id),0]
        #         label2id[onetag][1] += 1
        #     temp.append({'text':text, 'catgy':[ label2id[onetag][0] for onetag in tags ]})

        # 使用筛选后的label，所有的label出现次数均大于1000，共43个

        # 使用筛选后的数据 46407 条
        label2id = {".net": 0, "c#": 1, "database": 2, "mysql": 3, "sql-server": 4, "sql-server-2005": 5, "php": 6, "objective-c": 7, "iphone": 8, "web-services": 9, "windows": 10, "python": 11, "sql": 12, "css": 13, "html": 14, "asp.net": 15, "regex": 16, "c++": 17, "javascript": 18, "vb.net": 19, "visual-studio": 20, "asp.net-mvc": 21, "string": 22, "winforms": 23, "ajax": 24, "linq-to-sql": 25, "linq": 26, "performance": 27, "c": 28, "java": 29, "wpf": 30, "oop": 31, "wcf": 32, "multithreading": 33, "ruby": 34, "ruby-on-rails": 35, "tsql": 36, "jquery": 37, "xml": 38, "arrays": 39, "django": 40, "android": 41, "cocoa-touch": 42,}
        temp = []
        for i in range(1,len(data)):

            tags = data[i][3]
            # 分隔符是###,为了处理c#,需要倒转再倒转
            tags = tags[::-1].split("###")[:-1]
            if len(tags) == 1:
                continue # 跳过只有一个标签的数据
            tags = [ onetag[::-1] for onetag in tags ]

            used_tags = []
            for onetag in tags:
                if onetag in label2id.keys():
                    used_tags.append(onetag)
            if len(used_tags) <= 1:
                continue # 跳过筛选后只有一个标签的数据

            text = data[i][1] + ' ' + data[i][2]
            text = clean_str(text)

            temp.append({'text':text, 'catgy':[ label2id[onetag] for onetag in used_tags ]})


        train = temp[:40000]
        test  = temp[40000:]

        # 看看数据的label分布
        # [678, 1485, 212, 650, 290, 140, 931, 451, 523, 90, 126, 280, 580, 363, 612, 675, 294, 293, 1015, 151, 116, 205,         167, 208, 270, 107, 158, 131, 194, 424, 186, 116, 88, 149, 195, 229, 137, 860, 249, 201, 131, 156, 109]
        # [5768, 10494, 1500, 3396, 2825, 1129, 4950, 1830, 2416, 883, 830, 1530, 4087, 2004, 3576, 4833, 1683, 1621, 5710, 1357, 817, 1656, 1089, 1474, 1759, 895, 1244, 1143, 960, 2242, 900, 811, 680, 1034, 1149, 1205, 1103, 4658, 1497, 1359, 790, 380, 1089]
        #
        # count = [0 for i in range(43)]
        # for i in test:
        #     for j in i['catgy']:
        #         count[j] += 1
        # print(count)
        #
        # count = [0 for i in range(43)]
        # for i in train:
        #     for j in i['catgy']:
        #         count[j] += 1
        # print(count)


        trn_sents, Y_trn, Y_trn_o = load_data_and_labels(train,label2id)
        tst_sents, Y_tst, Y_tst_o = load_data_and_labels(test,label2id)


        trn_sents_padded = pad_sentences(trn_sents, max_length=sents_max_len)
        tst_sents_padded = pad_sentences(tst_sents, max_length=sents_max_len)

        vocabulary, vocabulary_inv, vocabulary_count = build_vocab(trn_sents_padded + tst_sents_padded, vocab_size=vocab_size)

        X_trn = build_input_data(trn_sents_padded, vocabulary)
        X_tst = build_input_data(tst_sents_padded, vocabulary)

        r =  {'train': (X_trn, Y_trn, Y_trn_o),
                'test' : (X_tst, Y_tst, Y_tst_o),
               'vocab' : (vocabulary, vocabulary_inv, vocabulary_count),
                'embed': load_word2vec(datapath ,'glove', vocabulary_inv, 300),
                'train_points': [(X_trn[i], Y_trn[i], Y_trn_o[i], i)  for i in range(len(Y_trn_o))],
                'test_points' : [(X_tst[i], Y_tst[i], Y_tst_o[i], i)  for i in range(len(Y_tst_o))]
                }
        with open(os.path.join(datapath, 'stackLoaded.pkl'), 'wb') as f:
            pickle.dump(r,f)
        return r


    def load_aapd(self, datapath, sents_max_len=300, vocab_size=50000):

        # 读取已缓存数据
        if os.path.exists(os.path.join(datapath, 'aapdLoaded.pkl')):
            with open(os.path.join(datapath, 'aapdLoaded.pkl'), 'rb') as f:
                return pickle.load(f)

        # [{'text':"...", 'catgy':['cat1','cat2',...] },]
        file_names = ['aapd_doc','aapd_tag']
        path = os.path.join(datapath, file_names[0])
        assert os.path.exists(path)
        with open(path) as f1:
            docs = f1.readlines()

        path = os.path.join(datapath, file_names[1])
        assert os.path.exists(path)
        with open(path) as f1:
            tags = f1.readlines()

        assert len(docs) == len(tags)

        label2id = dict()  # {name: [id,count]}
        data = []
        for text,tag in zip(docs,tags):
            tag = tag.strip().split()
            if len(tag) == 1:
                continue
            for onetag in tag:
                if onetag not in label2id.keys():
                    label2id[onetag] = [len(label2id),0]
                label2id[onetag][1] += 1

            text = clean_str(text)
            data.append({'text': text, 'catgy': [label2id[onetag][0] for onetag in tag]})

        train = data[:30000]
        test = data[30000:]

        # 看看数据的label分布
        # [1108, 261, 873, 9580, 9580, 367, 1523, 1594, 2762, 528, 271, 1500, 707, 4408, 1715, 1268, 2645, 1695, 515, 552, 1668, 534, 370, 2229, 1856, 1224, 706, 1881, 2739, 660, 1118, 2283, 391, 1967, 890, 261, 571, 520, 198, 931, 579, 216, 486, 231, 635, 474, 742, 378, 207, 399, 479, 479, 198, 222]
        # [117, 21, 83, 964, 964, 28, 165, 155, 270, 59, 24, 168, 78, 454, 183, 107, 282, 180, 44, 50, 158, 49, 50, 195, 175, 125, 58, 173, 256, 60, 102, 224, 42, 187, 95, 26, 70, 62, 21, 97, 47, 12, 45, 21, 68, 48, 80, 32, 22, 34, 46, 46, 24, 22]
        # count = [0 for i in range(len(label2id))]
        # for i in train:
        #     for j in i['catgy']:
        #         count[j] += 1
        # print(count)
        # count = [0 for i in range(len(label2id))]
        # for i in test:
        #     for j in i['catgy']:
        #         count[j] += 1
        # print(count)

        trn_sents, Y_trn, Y_trn_o = load_data_and_labels(train, label2id)
        tst_sents, Y_tst, Y_tst_o = load_data_and_labels(test, label2id)

        trn_sents_padded = pad_sentences(trn_sents, max_length=sents_max_len)
        tst_sents_padded = pad_sentences(tst_sents, max_length=sents_max_len)

        vocabulary, vocabulary_inv, vocabulary_count = build_vocab(trn_sents_padded + tst_sents_padded,
                                                                   vocab_size=vocab_size)

        X_trn = build_input_data(trn_sents_padded, vocabulary)
        X_tst = build_input_data(tst_sents_padded, vocabulary)

        r = {'train': (X_trn, Y_trn, Y_trn_o),
             'test': (X_tst, Y_tst, Y_tst_o),
             'vocab': (vocabulary, vocabulary_inv, vocabulary_count),
             'embed': load_word2vec(datapath, 'glove', vocabulary_inv, 300),
             'train_points': [(X_trn[i], Y_trn[i], Y_trn_o[i], i) for i in range(len(Y_trn_o))],
             'test_points': [(X_tst[i], Y_tst[i], Y_tst_o[i], i) for i in range(len(Y_tst_o))]
             }
        with open(os.path.join(datapath, 'aapdLoaded.pkl'), 'wb') as f:
            pickle.dump(r, f)
        return r



    def load_yahoo(self, datapath, pretrained, word_dim = 100, answer_count = 5):

        trainpath = os.path.join(datapath, 'train.txt')
        testpath = os.path.join(datapath, 'test.txt')

        train_data = []
        with open(trainpath) as f:
            for line in f:
                content = line.strip().split('#')
                if len(content) != 3:
                    continue
                if int(content[2]) != 0 and int(content[2]) != 1:
                    continue
                train_data.append((content[0], content[1], content[2]))

        test_data = []
        with open(testpath) as f:
            for line in f:
                content = line.strip().split('#')
                if len(content) != 3:
                    continue
                if int(content[2]) != 0 and int(content[2]) != 1:
                    print(content[2])
                    continue
                test_data.append((content[0], content[1], content[2]))

        dico_words_train = self.word_mapping(train_data)[0]

        all_embedding = False

        dico_words, word_to_id, id_to_word = augment_with_pretrained(
            dico_words_train.copy(),
            pretrained,
            list(itertools.chain.from_iterable(
                [[w.lower() for w in (s[0] + ' ' + s[1]).split()] for s in test_data]  # 样本包含问题和答案
            )
            ) if not all_embedding else None)

        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_data)

        train_data_final = prepare_dataset(train_data, word_to_id, tag_to_id)
        test_val_data = prepare_dataset(test_data, word_to_id, tag_to_id)

        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))


        # If the words in the training set do not exist in the pre-trained word vector,
        # replace them with a randomly generated uniformly distributed vector
        not_exist = 0
        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]
            else:
                not_exist += 1
        #print(" %d new words" % (not_exist))

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }

        npr = np.random.RandomState(seed = 0)
        data_index = npr.permutation(int(len(test_val_data) / answer_count))

        val_data_final = [test_val_data[no] for sampleID in data_index[:len(data_index) // 2]
                          for no in range(sampleID * answer_count, sampleID * answer_count + 5)]
        test_data_final = [test_val_data[no] for sampleID in data_index[len(data_index) // 2:]
                          for no in range(sampleID * answer_count, sampleID * answer_count + 5)]


        return train_data_final, val_data_final, test_data_final, mappings
