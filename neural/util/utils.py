from __future__ import print_function
import torch
import os
import re
import codecs
import copy
import numpy as np

import random
random.seed(0)

from collections import Counter


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file

    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def tag_mapping(dataset):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [s[2] for s in dataset]
    dico = Counter(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(dataset, word_to_id, tag_to_id):

    def f(x): return x.lower()

    data = []
    for s in dataset:
        str_words_q = [w for w in s[0].split()]
        str_words_a = [w for w in s[1].split()]
        words_q = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']for w in str_words_q]
        words_a = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']for w in str_words_a]
        tag = tag_to_id[s[2]]
        data.append({
            'str_words_q': str_words_q,
            'str_words_a': str_words_a,
            'words_q': words_q,
            'words_a': words_a,
            'tag': tag,
        })
    return data


def pad_seq(seq, max_length, PAD_token=0):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def create_batches(dataset, batch_size, order='no'):
# dataset should be n*datapoints
# one datapoint is
# [
#  [] # X   --> 单词的索引        | type: ndarray |  size: 1* 500
#  [] # Y   --> vector of 0,1    | type: csr_matrix
#  [] # Y_o --> labels, such as [1,2,3] for one x
# ]
    dataset = copy.deepcopy(dataset)
    if order == 'sort':
        # newdata.sort(key=lambda x: len(x['words']))
        assert('not finished this function')

    elif order == 'random':
        random.shuffle(dataset)

    else: #不改变原始顺序
        pass

    batches = []
    num_batches = np.ceil(len(dataset) / float(batch_size)).astype('int')

    for i in range(num_batches):
        batch_data = dataset[(i * batch_size):min(len(dataset), (i + 1) * batch_size)]
        batch_data = {'data_points':batch_data,
                      'data_numpy' :(   np.vstack( [i[0] for i in batch_data] ), # X
                                        np.vstack([i[1].A.astype(int) for i in batch_data]), #Y
                                        [i[2] for i in batch_data]  )
                      }
        batches.append(batch_data)

    return batches


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu) ** 2 / (2 * sigma ** 2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu) ** 2 / (2 * torch.exp(logsigma) ** 2)


def bayes_loss_function(l_pw, l_qw, l_likelihood, n_batches, batch_size):
    return ((1. / n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]