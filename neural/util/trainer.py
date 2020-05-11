from __future__ import print_function
from torch.autograd import Variable
import time
from .evaluator import Evaluator
from .utils import *
import sys
import os
import numpy as np

import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, model, result_path, model_name, tag_to_id,
                 eval_every=1, usecuda=True, answer_count = 5, cuda_device = 0):
        self.model = model
        self.eval_every = eval_every
        self.model_name = os.path.join(result_path, model_name)
        self._model_name = model_name
        self.usecuda = usecuda
        self.tagset_size = len(tag_to_id)
        self.lossfunc = nn.CrossEntropyLoss()
        self.cuda_device = cuda_device
        self.evaluator = Evaluator(result_path, model_name,
                                   answer_count = answer_count, cuda_device = self.cuda_device).evaluate_rank


    ##############################################################################
    ################# training only by supervised learning        ################
    def train_supervisedLearning(self, num_epochs, train_data, val_data, test_data, learning_rate, checkpoint_folder='.',
                     plot_every=2, batch_size=50):

        losses = []
        lossD = 0.0
        best_test_mrr = -1.0


        self.model.train(True)
        print('********Training Start*******')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(1, num_epochs + 1):
            t = time.time()
            count = 0
            batch_count = 0

            train_batches = create_batches(train_data, batch_size=batch_size, order='random')

            for i, index in enumerate(np.random.permutation(len(train_batches))):

                data = train_batches[index]
                self.model.zero_grad()

                words_q = data['words_q']
                words_a = data['words_a']
                tags = data['tags']

                words_q = Variable(torch.LongTensor(words_q)).cuda(self.cuda_device)
                words_a = Variable(torch.LongTensor(words_a)).cuda(self.cuda_device)
                tags = Variable(torch.LongTensor(tags)).cuda(self.cuda_device)

                wordslen_q = data['wordslen_q']
                wordslen_a = data['wordslen_a']

                if self._model_name in ['BiLSTM']:
                    output = self.model(words_q, words_a, tags, wordslen_q, wordslen_a, usecuda=self.usecuda)
                elif self._model_name in ['CNN']:
                    output = self.model(words_q, words_a, tags, usecuda=self.usecuda)

                loss = self.lossfunc(output, tags)
                lossD += loss.item() / len(wordslen_q)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                count += 1
                batch_count += len(wordslen_q)

                if count % plot_every == 0:
                    lossD /= plot_every
                    #print(batch_count, ': ', lossD)
                    if losses == []:
                        losses.append(lossD)
                    losses.append(lossD)
                    lossD = 0.0

            ####################################### Validation ###########################################
            if epoch % self.eval_every == 0:

                best_test_mrr, new_test_mrr, save = self.evaluator(self.model, val_data, best_test_mrr, model_name = self._model_name)

                print('*'*50)
                print("Validation：best_mrr：%f new_mrr:%f " % (best_test_mrr, new_test_mrr))

                if save:
                    print('Saving Best Weights')
                    torch.save(self.model, os.path.join(self.model_name, checkpoint_folder, 'modelweights'))

                sys.stdout.flush()

            print('Epoch %d Complete: Time Taken %d' % (epoch, time.time() - t))

        _, test_mrr, _ = self.evaluator(torch.load(os.path.join(self.model_name, checkpoint_folder, 'modelweights')),
                                        test_data,
                                        model_name=self._model_name)

        return test_mrr


