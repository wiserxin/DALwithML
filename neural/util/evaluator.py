import os
import codecs
import torch
import numpy as np
from .utils import *
import torch
from torch.autograd import Variable
from multiprocessing import Pool

# np.set_printoptions(threshold=np.inf)

#
# class Evaluator(object):
#     def __init__(self, result_path, model_name, usecuda=True, answer_count = 5, cuda_device = 0):
#         self.result_path = result_path
#         self.model_name = model_name
#         self.usecuda = usecuda
#         self.answer_count = answer_count
#         self.cuda_device = cuda_device
#
#
#     def evaluate_rank(self, model, dataset, best_mrr = 0.0, model_name='CNN'):
#
#         model.train(False)
#
#         RR = []
#
#         save = False
#
#         #get RR value for a prediction
#         def getRR(predicted, ground_truth):
#             zipped = zip(predicted, ground_truth)
#             temp = sorted(zipped, key=lambda p: p[0], reverse=True)
#             _, ground_truth_ids = zip(*temp)
#             for i in range(len(ground_truth_ids)):
#                 if ground_truth_ids[i] == 1:
#                     return (1.0 / float(i + 1))
#
#         data_batches = create_batches(dataset, batch_size = 1000, order='no')
#
#         for data in data_batches:
#
#             predicted_scores = []
#             predicted_ids = []
#             ground_truth_ids = []
#
#             words_q = data['words_q']  # [[1,2,3,4,9],[4,3,5,0,0]]
#             words_a = data['words_a']  # [[1,2,3,4,9],[4,3,5,0,0]]
#
#             words_q = Variable(torch.LongTensor(words_q)).cuda(self.cuda_device)
#             words_a = Variable(torch.LongTensor(words_a)).cuda(self.cuda_device)
#
#             # 句子长度
#             wordslen_q = data['wordslen_q']
#             wordslen_a = data['wordslen_a']
#
#             if model_name == 'BiLSTM':
#                 score, out = model.predict(words_q, words_a,
#                                            wordslen_q, wordslen_a,
#                                            usecuda=self.usecuda)
#             elif model_name == 'CNN':
#                 score, out = model.predict(words_q, words_a,
#                                        usecuda=self.usecuda)
#
#
#             ground_truth_ids.extend(data['tags'])
#             predicted_ids.extend(out)
#             predicted_scores.extend(score)
#
#
#             for i in range(len(predicted_scores)):
#                 if int(predicted_ids[i]) == 0:
#                     predicted_scores[i] = 1 - predicted_scores[i]
#
#
#             for i in range(int(len(data["words_q"]) / self.answer_count)):
#                 temp_1 = []
#                 temp_2 = []
#                 for j in range(self.answer_count):
#                     temp_1.append(predicted_scores[i * self.answer_count + j])
#                     temp_2.append(ground_truth_ids[i * self.answer_count + j])
#                 RR.append(getRR(temp_1, temp_2))
#
#         # MRR
#         new_mrr = round(sum(RR) / len(RR), 4)
#
#         if new_mrr > best_mrr:
#             best_mrr = new_mrr
#             save = True
#
#         model.train(True)
#
#         return best_mrr, new_mrr, save


class Evaluator(object):
    def __init__(self, result_path, model_name, usecuda=True, top_k=50, batch_size=128, cuda_device=0):

        self.model_name = model_name
        self.usecuda = usecuda
        self.top_k = top_k
        self.batch_size = batch_size
        self.cuda_device = cuda_device


    def precision_at_k(self, r, k):
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def dcg_at_k(self, r, k):
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    def ndcg_at_k(self, r, k):
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k) / dcg_max

    def get_result_ori(self, args):
    # args --> (y_pred, y_true)
    # 比较慢

        (y_pred, y_true) = args
        pred_topk_index = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)[:self.top_k]
        # print("ori:",pred_topk_index)
        pos_index = set([k for k, v in enumerate(y_true) if v == 1])

        r = [1 if k in pos_index else 0 for k in pred_topk_index[:self.top_k]]
        # print("ori:",r)

        # p_1 = self.precision_at_k(r, 1)
        # p_3 = self.precision_at_k(r, 3)
        # p_5 = self.precision_at_k(r, 5)
        # ndcg_1 = self.ndcg_at_k(r, 1)
        # ndcg_3 = self.ndcg_at_k(r, 3)
        # ndcg_5 = self.ndcg_at_k(r, 5)
        # return np.array([p_1, p_3, p_5, ndcg_1, ndcg_3, ndcg_5])

        ndcg_5 = self.ndcg_at_k(r, 5)
        return np.array([ndcg_5])


    def get_result(self, args):
    # args --> (y_pred, y_true)
    # np.argpartition : O(n)

        (y_pred, y_true) = args
        y_pred = y_pred.data.numpy()
        pred_topk_index = np.argpartition(y_pred, -self.top_k)[-self.top_k:] # 不分先后的选出排名靠前者
        pred_topk_index = sorted([j for j in pred_topk_index], key=lambda i: y_pred[i], reverse=True) # 分先后的选出来
        # print('better:',pred_topk_index)
        pos_index = set([k for k, v in enumerate(y_true) if v == 1])

        r = [1 if k in pos_index else 0 for k in pred_topk_index]
        # print("better:",r)

        # p_1 = self.precision_at_k(r, 1)
        # p_3 = self.precision_at_k(r, 3)
        # p_5 = self.precision_at_k(r, 5)
        # ndcg_1 = self.ndcg_at_k(r, 1)
        # ndcg_3 = self.ndcg_at_k(r, 3)
        # ndcg_5 = self.ndcg_at_k(r, 5)
        # return np.array([p_1, p_3, p_5, ndcg_1, ndcg_3, ndcg_5])

        ndcg_5 = self.ndcg_at_k(r, 5)
        return np.array([ndcg_5])


    def evaluate(self, model, dataset, best_result = 0.0, model_name='CNN', multi_progress_num= 4):
        # for data - matrix
        # dataset --> [(X, Y, Y_o), ... , ( , , )]
        # X --> np.ndarray
        # for example:  Y[0]-->[0,1,1,0]  Y_o[0]-->[1,2]


        model.train(False)

        results = []
        train_batches = create_batches(dataset, batch_size=self.batch_size, order='no')

        for i, batch_data in enumerate(train_batches):
            batch_data = batch_data['data_numpy']
            X_batch, Y_batch, _ = batch_data
            X_batch = Variable(torch.from_numpy(X_batch).long())
            X_batch = X_batch.cuda(self.cuda_device) if self.usecuda else X_batch

            Y_batch = Variable(torch.from_numpy(Y_batch).int())

            Y_pred   = model(X_batch).cpu()
            if Y_pred.shape[1] >= Y_batch.shape[1]:
                Y_pred = Y_pred[:, :Y_batch.shape[1]]

            # 使用多线程 evaluate , 有 BUG
            # pool = Pool(multi_progress_num)
            # pool_result_map = pool.map(self.get_result, zip(list(Y_pred.detach().numpy()), list(Y_batch.detach().numpy())))
            # pool.terminate()
            # results.extend(list(pool_result_map))

            # 非多线程 evaluate
            results.extend(list(map(self.get_result, zip(list(Y_pred), list(Y_batch)) )))

            print("\rEvaluating: {}/{} ({:.1f}%)".format(i,len(train_batches),i*100/len(train_batches)), end=' ')

        results = np.array(list(results))
        tst_result = list(np.mean(results, 0))[0]

        save = tst_result > best_result
        best_result = tst_result if tst_result>best_result else best_result

        model.train(True)

        return best_result, tst_result, save

    def evaluate_for_datapoints(self, model, dataset, best_result = 0.0, model_name='CNN', multi_progress_num= 4):

        model.train(False)
        batchs = create_batches(dataset,self.batch_size)
        results = []

        for i,batch_data in enumerate(batchs):
            batch_data = batch_data['dat_numpy']
            X_batch = batch_data[0]
            Y_batch = batch_data[1]

            X_batch = torch.from_numpy(X_batch).long()
            X_batch = X_batch.cuda() if self.usecuda else X_batch
            Y_batch = Variable(torch.from_numpy(Y_batch).int())

            Y_pred = model(X_batch).cpu()
            if Y_pred.shape[1] >= Y_batch.shape[1]:
                Y_pred = Y_pred[:, :Y_batch.shape[1]]

            # 使用多线程 evaluate , 有 BUG
            # pool = Pool(multi_progress_num)
            # pool_result_map = pool.map(self.get_result, zip(list(Y_pred.detach().numpy()), list(Y_batch.detach().numpy())))
            # pool.terminate()
            # results.extend(list(pool_result_map))

            # 非多线程 evaluate
            results.extend(list(map(self.get_result, zip(list(Y_pred), list(Y_batch)))))

            print("\rEvaluating: {}/{} ({:.1f}%)".format(i, len(batchs), i * 100 / len(batchs)), end=' ')

        results = np.array(list(results))
        tst_result = list(np.mean(results, 0))[0]

        save = tst_result > best_result
        best_result = tst_result if tst_result > best_result else best_result

        model.train(True)

        return best_result, tst_result, save