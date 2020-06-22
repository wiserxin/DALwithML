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

    def __init__(self, model, result_path, model_name,
                 eval_begin=1, eval_every=1, top_k=15, usecuda=True, cuda_device = 0, ndcg_num=5):
        self.model = model
        self.eval_begin = eval_begin
        self.eval_every = eval_every
        self.model_name = os.path.join(result_path, model_name)
        self._model_name = model_name
        self.usecuda = usecuda
        self.lossfunc = nn.MultiLabelSoftMarginLoss()
        self.cuda_device = cuda_device
        self.evaluator = Evaluator(result_path, model_name,top_k=top_k,
                                   cuda_device = self.cuda_device,ndcg_num=ndcg_num).evaluate_with_datapoints_F1


    ##############################################################################
    ################# training only by supervised learning        ################
    def train_supervisedLearning(self, num_epochs, train_data, val_data, learning_rate, checkpoint_path='.',
                     plot_every=2, batch_size=50):

        losses = []
        lossD = 0.0
        # best_test_ndcg5 = 0.0
        best_eval = 0.0


        self.model.train(True)
        print('********Training Start*******')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(1, num_epochs + 1):
            t = time.time()
            count = 0
            batch_count = 0

            train_batches = create_batches(train_data, batch_size=batch_size, order='random')

            for i, batch_data in enumerate(np.random.permutation(train_batches)):

                batch_data = batch_data['data_numpy']
                self.model.zero_grad()

                # X , Y, Y_o, _ = batch_data
                X = batch_data[0]
                Y = batch_data[1]
                Y_o = batch_data[2]

                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
                Y = Variable(torch.from_numpy(Y).float()).cuda(self.cuda_device)


                if self._model_name in ['BiLSTM']:
                    # output = self.model(words_q, words_a, tags, wordslen_q, wordslen_a, usecuda=self.usecuda)
                    pass
                elif self._model_name in ['CNN']:
                    output = self.model(X, usecuda=self.usecuda)

                # print('output:',output.size(),'\n',output)
                # print('Y:',Y.size(),'\n',Y)
                # assert False

                loss = self.lossfunc(output, Y )
                lossD += loss.item() / len(Y_o)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                count += 1
                batch_count += len(Y_o)

                torch.cuda.empty_cache()

                if count % plot_every == 0:
                    lossD /= plot_every
                    #print(batch_count, ': ', lossD)
                    if losses == []:
                        losses.append(lossD)
                    losses.append(lossD)
                    lossD = 0.0

            ####################################### Validation ###########################################
            if (epoch >= self.eval_begin) and (epoch % self.eval_every == 0):

                # best_test_ndcg5, new_test_ndcg5, save = self.evaluator(self.model, val_data, best_test_ndcg5, model_name = self._model_name)
                best_eval, new_eval, save = self.evaluator(self.model, val_data, best_eval, model_name = self._model_name)
                print('*'*50)
                print("Validation：best：{}   new: {} ".format(best_eval, new_eval))

                if save:
                    print('Saving Best Weights')
                    torch.save(self.model, os.path.join(checkpoint_path, 'modelweights'))

                sys.stdout.flush()

            print('Epoch %d Complete: Time Taken %d' % (epoch, time.time() - t))

        # 需要注释掉？
        # _, test_mrr, _ = self.evaluator(torch.load(os.path.join(self.model_name, checkpoint_folder, 'modelweights')),
        #                                 test_data,
        #                                 model_name=self._model_name)

        # return best_test_ndcg5
        return best_F1


