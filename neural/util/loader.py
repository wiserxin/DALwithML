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
            m = max(row_idx) + 1
            n = max(col_idx) + 1
            Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
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

        return {'train': (X_trn, Y_trn, Y_trn_o),
                'test' : (X_tst, Y_tst, Y_tst_o),
               'vocab' : (vocabulary, vocabulary_inv, vocabulary_count),
                'embed': load_word2vec(datapath ,'glove', vocabulary_inv, 300),
                'train_points': [(X_trn[i], Y_trn[i], Y_trn_o[i])  for i in range(len(Y_trn_o))],
                'test_points' : [(X_tst[i], Y_tst[i], Y_tst_o[i])  for i in range(len(Y_tst_o))]
                }


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

        def load_data_and_labels(data):
            x_text = [doc['text'] for doc in data]
            x_text = [s.split(" ") for s in x_text]
            labels = [doc['catgy'] for doc in data]
            row_idx, col_idx, val_idx = [], [], []
            for i in range(len(labels)):
                l_list = list(set(labels[i]))  # remove duplicate cateories to avoid double count
                for y,_ in enumerate(l_list):
                    row_idx.append(i)
                    col_idx.append(y)
                    val_idx.append(1)
            m = max(row_idx) + 1
            n = max(col_idx) + 1
            Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
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

        # [{'text':"...", 'catgy':['','',...] },]
        file_names = ['train_texts.txt','train_labels.txt','test_texts.txt','test_labels.txt']

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
                        train.append({'text':text, 'catgy':labels})


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
                        test.append({'text':text, 'catgy':labels})


        trn_sents, Y_trn, Y_trn_o = load_data_and_labels(train)
        tst_sents, Y_tst, Y_tst_o = load_data_and_labels(test)

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
                'train_points': [(X_trn[i], Y_trn[i], Y_trn_o[i])  for i in range(len(Y_trn_o))],
                'test_points' : [(X_tst[i], Y_tst[i], Y_tst_o[i])  for i in range(len(Y_tst_o))]
                }
        with open(os.path.join(datapath, 'rcv2Loaded.pkl'), 'wb') as f:
            pickle.dump(r,f)
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
