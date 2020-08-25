import torch
from torch.autograd import Variable
import numpy as np
import time
from scipy import stats
from scipy.spatial import distance_matrix
from neural.util.utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from .unsupervised import *
from sklearn.cluster import KMeans


class Acquisition(object):
    def __init__(self, train_data,
                 seed=0,
                 usecuda=True,
                 cuda_device=0,
                 batch_size=1000,
                 cal_Aleatoric_uncertainty=False,
                 submodular_k=4,
                 target_size = 2):
        self.train_data = train_data
        self.document_num = len(train_data)
        self.train_index = set()  #the index of the labeled samples
        self.pseudo_train_data = []
        self.random_seed = seed
        self.npr = np.random.RandomState(seed)
        self.usecuda = usecuda
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.cal_Aleatoric_uncertainty = cal_Aleatoric_uncertainty
        self.submodular_k = submodular_k
        self.savedData = list()
        self.label_count = np.zeros(target_size) # 存储train_index中的数据，其中涉及到的label的计数
        # for i in range(len(train_data)):
        #     print(i,train_data[i][3])

    def eval_acquire_rcv2_points(self,index_arry):
        # 统计index中的label数量

        def wanderDatalabels(datapoints):
            Y = {}
            for point in datapoints:
                Y_o = point[2]
                id = point[3]
                for label in Y_o:
                    if label in Y.keys():
                        Y[label] += 1
                    else:
                        Y[label] = 1
            return Y
        theList = [self.train_data[i] for i in index_arry]
        theDict = wanderDatalabels(theList)
        result = [(theDict[i] if i in theDict.keys() else 0) for i in range(103)]
        return result

    def update_train_index(self,acquired_set):
        # 注意调用时应保证 先 self.savedData.append , 再 update_train_index
        # 更新 label_count 和 train_index
        temp = []
        for i in acquired_set:
            temp.append(np.array(self.train_data[i][1].todense()))
        train_Y = np.squeeze(np.stack(temp))
        # print(train_Y.shape)
        each_label_count = np.sum(train_Y, axis=0)
        self.label_count = self.label_count + each_label_count
        # print(self.label_count)
        # print(self.label_count.shape)
        # assert False

        self.train_index.update(acquired_set)

        if len(self.savedData) == 0:
            self.savedData.append({"train_index":list(self.train_index)})
        else:
            self.savedData[-1]["train_index"]=list(self.train_index)


    def get_random(self, data, acquire_num, returned=False):

        random_indices = self.npr.permutation(self.document_num)
        # random_indices = range(self.document_num) # 测试专用 ，正式使用时应注释掉

        sample_indices = set()
        i = 0
        while len(sample_indices) < acquire_num :
            if random_indices[i] not in self.train_index:
                sample_indices.add(random_indices[i])
            i += 1
        if not returned:
            self.update_train_index(sample_indices)
            # print('now type1:type2 = {}:{}'.format(
            #                     sum([1 if i<200 else 0 for i in self.train_index ]) ,
            #                     sum([1 if i >= 200 else 0 for i in self.train_index]),
            #       ))
        else:
            return sample_indices

    def get_DAL(self, dataset, model_path, acquire_document_num,
                nsamp=100,
                model_name='',
                returned=False,
                ):

        model = torch.load(model_path)
        model.train(True) # 保持 dropout 开启
        tm = time.time()

        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

        print('DAL: preparing batch data',end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []

        for iter_batch,data in enumerate(data_batches):
            print('\rDAL acquire batch {}/{}'.format(iter_batch,len(data_batches)),end='')

            batch_data_numpy  = data['data_numpy']
            batch_data_points = data['data_points']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())


            tag_arr = []
            score_arr = []
            real_tag_arr = []
            # sigma_total = torch.zeros((nsamp, words_q.size(0))) # ???

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    # output = model(words_q, words_a, wordslen_q, wordslen_a)
                    pass
                elif model_name == 'CNN':
                    output = model(X)

                score = torch.sigmoid(output).data.cpu().numpy().tolist()
                # score = torch.softmax(output,dim=1).data.cpu().numpy().tolist() # 测试softmax
                # score = F.logsigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))


            # print("score_arr:",len(score_arr),len(score_arr[0]),len(score_arr[0][0]))

            # new_score_seq = np.array(score_arr).transpose(0, 1).tolist()
            new_score_seq = []  # size: btach_size * nsample * nlabel
            for m in range(len(Y)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))

            for index, item in enumerate(new_score_seq):
                # shape: batch_size * nsample * nlabel


                def rankedList(rList):
                    rList = np.array(rList)
                    gain = 2 ** rList - 1
                    discounts = np.log2(np.arange(len(rList)) + 2)
                    return np.sum(gain / discounts)

                tp1 = item # shape: nsample * nlabel
                item = np.transpose(np.array(item)).tolist() # shape: labels * nsample


                dList = []
                for i in range(len(tp1)):
                    rL = sorted(tp1[i], reverse=True)
                    dList.append(rankedList(rL))

                # t = np.mean(2 ** np.array(item) - 1, axis=1)
                # rankedt = sorted(t.tolist(), reverse=True)
                # d = rankedList(rankedt)

                item_arr = np.array(item)

                t = np.mean(item_arr, axis=1)
                rankedt = np.transpose(item_arr[(-t).argsort()]).tolist()  # nsamp, 5

                dList2 = []
                for i in range(len(rankedt)):
                    dList2.append(rankedList(rankedt[i]))

                obj = {}
                obj["id"] = pt
                obj["el"] = np.mean(np.array(dList)) - np.mean(np.array(dList2))
                # obj["el"] = np.mean(np.array(dList))      # 0 测试 一些
                # obj["el"] = obj["el"] * np.sum(item)      # 1 测试增加labels维度是否有提升  -- 惨，前期=随机，中期略高于随机，后期与随机不相上下
                # obj["el"] = obj["el"]*(1+np.sum(np.array(item)>0.5))          # 2 测试labels映射到{0,1}是否有提升
                                                                                # 3 测试labels不sigmoid性能如何
                # obj["el"] = obj["el"]*np.sum( 1-abs(1-2*np.array(item)) )     # 4 测试 inconfidence * el |  inconfidence = 1-abs(2*score-1)
                # obj["el"] = obj["el"] + np.mean( 1-abs(1-2*np.array(item)) )  # 4.2 测试 inconfidence + el
                                                                                # 4.3 前5轮inconfidence,之后都是el
                                                                                # 4.4 动态inconfidence权重 el+1/sqrt(r)*IC

                if obj["el"] < 0:
                    print("elo error")
                    exit()

                _delt_arr.append(obj)
                pt += 1
        print()

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True) # 从大到小排序
        # _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=False)  # 从小到大排序 测试效果如何

        # with open( "DAL_dele_arr.txt" , 'a+' ) as f:
        #     temp_delt_arr = [ i["el"] for i in _delt_arr ]
        #     print( temp_delt_arr ,file=f)

        cur_indices = set()
        i = 0

        while len(cur_indices) < acquire_document_num:
            try:
                cur_indices.add(new_datapoints[_delt_arr[i]["id"]])
                i += 1
            except:
                print(acquire_document_num)
                print(i)
                print(type(new_datapoints),len(new_datapoints))
                print(new_datapoints[:10])
                print(_delt_arr[i])
                assert False

        if not returned:
            # print("DAL acquiring:",sorted(cur_indices))
            self.update_train_index(cur_indices)
            print('DAL time consuming： %d seconds:' % (time.time() - tm))
            # print('now type1:type2 = {}:{}'.format(
            #                     sum([1 if i<200 else 0 for i in self.train_index ]) ,
            #                     sum([1 if i >= 200 else 0 for i in self.train_index]),
            #       ))
        else:
            sorted_cur_indices = list(cur_indices)
            # sorted_cur_indices.sort()
            dataset_pool = []
            # for m in range(len(sorted_cur_indices)):
            #     item = dataset[sorted_cur_indices[m]]
            #     item["index"] = sorted_cur_indices[m]
            #     dataset_pool.append(item)

            return dataset_pool, cur_indices

        self.savedData.append( { "added_index":cur_indices,
                                 "index2id":{_index:p[3] for _index,p in enumerate(new_dataset)},
                                 "_delt_arr":_delt_arr } )


    def get_RS2HEL(self, dataset, model_path, acquire_document_num,
                nsamp=100,
                model_name='',
                returned=False,
                thisround=-1,
                ):

        model = torch.load(model_path)
        model.train(True) # 保持 dropout 开启
        tm = time.time()

        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

        print('RS2HEL: preparing batch data',end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []

        for iter_batch,data in enumerate(data_batches):
            print('\rRS2HEL acquire batch {}/{}'.format(iter_batch,len(data_batches)),end='')

            batch_data_numpy  = data['data_numpy']
            batch_data_points = data['data_points']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())


            tag_arr = []
            score_arr = []
            real_tag_arr = []
            # sigma_total = torch.zeros((nsamp, words_q.size(0))) # ???

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    # output = model(words_q, words_a, wordslen_q, wordslen_a)
                    pass
                elif model_name == 'CNN':
                    output = model(X)

                score = torch.sigmoid(output).data.cpu().numpy().tolist()
                # score = torch.softmax(output,dim=1).data.cpu().numpy().tolist() # 测试softmax
                # score = F.logsigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))


            # print("score_arr:",len(score_arr),len(score_arr[0]),len(score_arr[0][0]))

            # new_score_seq = np.array(score_arr).transpose(0, 1).tolist()
            new_score_seq = []  # size: btach_size * nsample * nlabel
            for m in range(len(Y)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))

            for index, item in enumerate(new_score_seq):
                # shape: batch_size * nsample * nlabel


                def rankedList(rList):
                    rList = np.array(rList)
                    gain = 2 ** rList - 1
                    discounts = np.log2(np.arange(len(rList)) + 2)
                    return np.sum(gain / discounts)

                tp1 = item # shape: nsample * nlabel
                item = np.transpose(np.array(item)).tolist() # shape: labels * nsample


                dList = []
                for i in range(len(tp1)):
                    rL = sorted(tp1[i], reverse=True)
                    dList.append(rankedList(rL))

                # t = np.mean(2 ** np.array(item) - 1, axis=1)
                # rankedt = sorted(t.tolist(), reverse=True)
                # d = rankedList(rankedt)

                item_arr = np.array(item)

                t = np.mean(item_arr, axis=1)
                rankedt = np.transpose(item_arr[(-t).argsort()]).tolist()  # nsamp, 5

                dList2 = []
                for i in range(len(rankedt)):
                    dList2.append(rankedList(rankedt[i]))

                obj = {}
                obj["id"] = pt
                obj["el"] = np.mean(np.array(dList)) - np.mean(np.array(dList2))

                if obj["el"] < 0:
                    print("elo error")
                    exit()

                _delt_arr.append(obj)
                pt += 1
        print()

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True) # 从大到小排序


        cur_indices = set()
        i = 0

        # 在前 4*acquire_document_num 中随机选 acquire_document_num 个 （未标记样本足够的话）
        sample_domin = min(4*acquire_document_num, len(new_dataset))
        sample_domin =  range( (max(3-thisround,0)+1) * sample_domin )
        sample_domin = self.npr.permutation(sample_domin)

        while len(cur_indices) < acquire_document_num:
            try:
                cur_indices.add(new_datapoints[_delt_arr[sample_domin[i]]["id"]])
                i += 1
            except:
                print(acquire_document_num)
                print(i)
                print(type(new_datapoints),len(new_datapoints))
                print(new_datapoints[:10])
                print(_delt_arr[i])
                assert False

        self.savedData.append({"added_index": cur_indices,
                               "index2id": {_index: p[3] for _index, p in enumerate(new_dataset)},
                               "_delt_arr": _delt_arr})

        if not returned:
            # print("DAL acquiring:",sorted(cur_indices))
            self.update_train_index(cur_indices)
            print('RS2HEL time consuming： %d seconds:' % (time.time() - tm))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, cur_indices


    def get_RKL(self, dataset, model_path, acquire_document_num,
                nsamp=100,
                model_name='',
                returned=False,
                rklNo = 4,      # 选取rkl策略，默认是rkl4
                density = False,#开启则挑选更稠密的区间的点
                thisround=-1,):

        def rankingLoss2(item):  # item = nsamples * labels
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0) > 0.5
            positiveItems = item_arr[:, overAllGroundTruth]
            negitiveItems = item_arr[:, overAllGroundTruth == 0]
            r = 0
            for column in range(positiveItems.shape[1]):
                temp = positiveItems[:, column]
                temp = negitiveItems.transpose() - temp
                r += np.sum((temp > 0) * temp)
            return r

        # def rankingLoss3(item, eachRankingLoss=True):
        #     # 设计的有问题，是否为正样本应按照大小排序，而不应用阈值
        #     # 使用阈值的话，会产生el error
        #     # item = nsamples * labels
        #     item_arr = np.array(item)
        #     if eachRankingLoss: # each时遇到异常样本时为 1
        #         r = []
        #         eachGroundTruth = item_arr > 0.5
        #         for i in range(item_arr.shape[0]):
        #             tn = (np.sum(eachGroundTruth[i] == 1))
        #             fn = (np.sum(eachGroundTruth[i] == 0))
        #             if tn == 0 or fn == 0:
        #                 r.append(1)
        #                 # pass
        #             else:
        #                 r.append(np.sum(item_arr[i] * eachGroundTruth[i]) / tn - np.sum(
        #                     item_arr[i] * (eachGroundTruth[i] == 0)) / fn)
        #         return np.array(r)
        #
        #     else: # overall时遇到异常样本返回-1
        #         overAllGroundTruth = np.mean(item_arr, axis=0) > 0.5
        #         tn = (np.sum(overAllGroundTruth == 1))
        #         fn = (np.sum(overAllGroundTruth == 0))
        #         if tn == 0 or fn == 0:
        #             return -1
        #         else:
        #             positiveItems = item_arr[:, overAllGroundTruth]
        #             negitiveItems = item_arr[:, overAllGroundTruth == 0]
        #             return np.mean(positiveItems) - np.mean(negitiveItems)

        def rankingLoss4(item):
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # each RL3
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            each_rl = np.mean((np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1)))

            # overall RL3
            sorted_item_arr = item_arr[:, overAllGroundTruth.argsort()]
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            overall_rl = np.mean(np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1))
            return each_rl - overall_rl

        def rankingLoss4_first_part(item):
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # each RL3
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            each_rl = np.mean((np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1)))

            return each_rl

        def rankingLoss5(item):
            # RKL5
            positive_num = 5# 超参数,意义为 期望有几个positive label
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            # print(overAllGroundTruth)

            # each RKL5
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, -2 * positive_num:-positive_num]
            each_rl = np.mean((np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1)))

            # overall RL5
            sorted_item_arr = item_arr[:, overAllGroundTruth.argsort()]
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, -2 * positive_num:-positive_num]
            overall_rl = np.mean(np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1))
            return 1+ each_rl - overall_rl# 应该更好的表征el的式子，不应使用 1+

        def rankingLoss6(item):
            return 2.0-rankingLoss5(item)

        def rankingLoss7(item,mod=1):
            weight = 1- self.label_count / np.sum(self.label_count)

            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            #     print(overAllGroundTruth)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            # print(overAllGroundTruth.size)
            # print(overAllGroundTruth.shape)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            temp = overAllGroundTruth.argsort()
            overAllGroundTruth = overAllGroundTruth[temp]
            weight = weight[temp]
            #     positive_labels = temp[-positive_num:]
            #     negitive_labels = temp[:-positive_num]

            # overall
            overAllGroundTruth_t = overAllGroundTruth * weight
            overall_loss = np.mean(weight[:-positive_num]) * np.mean(overAllGroundTruth_t[-positive_num:]) - \
                           np.mean(weight[-positive_num:]) * np.mean(overAllGroundTruth_t[:-positive_num])

            # each
            if mod == 1:
                sorted_item_arr_t = np.mean(np.sort(item_arr),axis=0)*weight
                each_loss = np.mean(weight[:-positive_num])*np.mean(sorted_item_arr_t[-positive_num:]) - \
                            np.mean(weight[-positive_num:])*np.mean(sorted_item_arr_t[:-positive_num])
            elif mod == 2:
                #         t = item_arr * weight
                pass

            return each_loss - overall_loss

        def rankingLoss9(item):
            # rkl9 = weight[trueLabels] * rkl4

            weight = 1 - self.label_count / np.sum(self.label_count)

            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # each RL3
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            each_rl = np.mean((np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1)))

            # overall RL3
            sorted_item_arr = item_arr[:, overAllGroundTruth.argsort()]
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            overall_rl = np.mean(np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1))
            return (each_rl - overall_rl) * np.sum(weight[overAllGroundTruth.argsort()[-positive_num:]])

        def meanMaxLoss(item):
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            #     print(overAllGroundTruth)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            # print(overAllGroundTruth.size)
            # print(overAllGroundTruth.shape)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            nlabels = item_arr.shape[1]
            M = np.eye(nlabels)
            M = 2 * M - 1  # 构建对角线为1 其余为-1 的矩阵

            # overall
            positive_labels = overAllGroundTruth.argsort()[-positive_num:]
            overall_loss = np.mean(np.sum(1 - M[positive_labels] * overAllGroundTruth, axis=1))
            #     print(overall_loss)

            # each
            sorted_item_arr = np.sort(item_arr)
            t = nlabels * 1 - np.dot(M[-positive_num:], item_arr.T)
            t = np.mean(t)
            each_loss = t

            return each_loss - overall_loss

        def entropyLoss(item,ee=False): # ee: entropy*EL ?
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            entropy_arr = -np.log2(item_arr) * item_arr - np.log2(1 - item_arr) * (1 - item_arr)
            overall_entropy = -np.log2(overAllGroundTruth) * overAllGroundTruth - np.log2(1 - overAllGroundTruth) * (
                        1 - overAllGroundTruth)
            overall_entropy = np.mean(overall_entropy)
            each_entropy = np.mean(entropy_arr)
            if ee:
                return overall_entropy* (overall_entropy - each_entropy)
            else:
                return overall_entropy - each_entropy

        def eentropyLoss(item):
            return entropyLoss(item,ee=True)

        def BALD(item):
            item_arr = np.array(item)
            pos_item_arr = item_arr > 0.5
            pos_item_code = list()
            for i in pos_item_arr:
                temp = 0
                for j in i:
                    temp = temp << 1
                    temp += j
                pos_item_code.append(temp)
            delt = stats.mode(pos_item_code)
            return len(item_arr) - delt[1][0] # 总采样次数 - 出现最多的模式的频次;得到的值越大,模型越不确定

        # 选取 rkl 策略
        rklDic = {2:rankingLoss2,
                  4:rankingLoss4,
                  5:rankingLoss5,
                  6:rankingLoss6,
                  7:rankingLoss7,
                  9:rankingLoss9,

                  '4fp':rankingLoss4_first_part,

                  'mml':meanMaxLoss,

                  'et':entropyLoss,
                  'ee':eentropyLoss,
                  'bald': BALD,
                  }
        rkl = rklDic[rklNo]
        print("RKL",rklNo,end="\t")



        model = torch.load(model_path)
        model.train(True) # 保持 dropout 开启
        tm = time.time()

        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

        # 防止死循环
        acquire_document_num = acquire_document_num if acquire_document_num <= len(new_datapoints) else len(new_datapoints)

        print('RKL: preparing batch data',end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []
        elo_count = 0

        for iter_batch,data in enumerate(data_batches):
            print('\rRKL acquire batch {}/{} elo num:{}'.format(iter_batch,len(data_batches),elo_count),end='')

            batch_data_numpy  = data['data_numpy']
            batch_data_points = data['data_points']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())

            score_arr = []

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    # output = model(words_q, words_a, wordslen_q, wordslen_a)
                    pass
                elif model_name == 'CNN':
                    output = model(X)

                score = torch.sigmoid(output).data.cpu().numpy().tolist()
                # score = torch.softmax(output,dim=1).data.cpu().numpy().tolist() # 测试softmax
                # score = F.logsigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))


            new_score_seq = []  # size: btach_size * nsample * nlabel
            for m in range(len(Y)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))

            for index, item in enumerate(new_score_seq):
                # new_xxx shape: batch_size * nsample * nlabel
                # item    shape: nsample * nlabel
                obj = {}
                obj["id"] = pt
                obj["el"] = rkl(item)

                if obj["el"] < -1e-10:
                    elo_count += 1
                    obj["el"] = -obj["el"]
                    # print("elo error:",obj["el"])
                    # exit()

                _delt_arr.append(obj)
                pt += 1


        if density: # 考虑在所有未标注点中，pt的密度
            print("\t density ing ...",end='')
            sim_matrix = self.getSimilarityMatrix(new_dataset,model_path,model_name)
            sim = np.mean(sim_matrix,axis=0)
            for pt,sim_t in enumerate(sim):
                assert (_delt_arr[pt]['id'] == pt)
                _delt_arr[pt]["el"] *= sim_t
        print()

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True) # 从大到小排序

        cur_indices = set()
        i = 0

        while len(cur_indices) < acquire_document_num:
            try:
                cur_indices.add(new_datapoints[_delt_arr[i]["id"]])
                i += 1
            except:
                print(acquire_document_num)
                print(i)
                print(type(new_datapoints),len(new_datapoints))
                print(new_datapoints[:10])
                print(_delt_arr[i])
                assert False

        self.savedData.append({"added_index": cur_indices,
                               "index2id": {_index: p[3] for _index, p in enumerate(new_dataset)},
                               "_delt_arr": _delt_arr})

        if not returned:
            self.update_train_index(cur_indices)
            print('RKL time consuming： %d seconds:' % (time.time() - tm))
        else:
            dataset_pool = []

            return dataset_pool, cur_indices



    def get_FEL(self, dataset, model_path, acquire_document_num,
                nsamp=100, model_name='', returned=False, thisround=-1,):
    # use F1 function calcualte the loss
    # just test it

        def F1Loss(item):
            from sklearn.metrics import f1_score
            baseLine = 0.5
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0) > baseLine

            overAllGroundTruth = np.tile(overAllGroundTruth, (item_arr.shape[0], 1))
            r = 0
            for am in ['micro', 'macro', 'weighted', 'samples']:
                r += (1.0 - f1_score(overAllGroundTruth, item_arr > baseLine, average=am))
            return r

        model = torch.load(model_path)
        model.train(True)  # 保持 dropout 开启

        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

        # 防止死循环
        acquire_document_num = acquire_document_num if acquire_document_num <= len(new_datapoints) else len(
            new_datapoints)

        print('FEL: preparing batch data', end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []

        for iter_batch, data in enumerate(data_batches):
            print('\rFEL acquire batch {}/{}'.format(iter_batch, len(data_batches)), end='')

            batch_data_numpy = data['data_numpy']
            batch_data_points = data['data_points']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())

            score_arr = []

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    # output = model(words_q, words_a, wordslen_q, wordslen_a)
                    pass
                elif model_name == 'CNN':
                    output = model(X)

                score = torch.sigmoid(output).data.cpu().numpy().tolist()
                # score = torch.softmax(output,dim=1).data.cpu().numpy().tolist() # 测试softmax
                # score = F.logsigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))

            new_score_seq = []  # size: btach_size * nsample * nlabel
            for m in range(len(Y)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))

            for index, item in enumerate(new_score_seq):
                # new_xxx shape: batch_size * nsample * nlabel
                # item    shape: nsample * nlabel
                obj = {}
                obj["id"] = pt
                obj["el"] = F1Loss(item)

                if obj["el"] < -1e-10:
                    print("elo error:", obj["el"])
                    exit()

                _delt_arr.append(obj)
                pt += 1
        print()

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True)  # 从大到小排序

        cur_indices = set()
        i = 0

        while len(cur_indices) < acquire_document_num:
            try:
                cur_indices.add(new_datapoints[_delt_arr[i]["id"]])
                i += 1
            except:
                print(acquire_document_num)
                print(i)
                print(type(new_datapoints), len(new_datapoints))
                print(new_datapoints[:10])
                print(_delt_arr[i])
                assert False

        if not returned:
            self.update_train_index(cur_indices)
        else:
            dataset_pool = []

            return dataset_pool, cur_indices

        self.savedData.append({"added_index": cur_indices,
                               "index2id": {_index: p[3] for _index, p in enumerate(new_dataset)},
                               "_delt_arr": _delt_arr})

    def get_FEL_RKL_ETL(self, dataset, model_path, acquire_document_num,
                        nsamp=100, model_name='', returned=False, thisround=-1,
                        combine_method = ""):
        def rankingLoss4(item):
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # each RL3
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            each_rl = np.mean((np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1)))

            # overall RL3
            sorted_item_arr = item_arr[:, overAllGroundTruth.argsort()]
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            overall_rl = np.mean(np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1))
            return each_rl - overall_rl

        def F1Loss(item):
            from sklearn.metrics import f1_score
            baseLine = 0.5
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0) > baseLine

            overAllGroundTruth = np.tile(overAllGroundTruth, (item_arr.shape[0], 1))
            r = 0
            for am in ['micro', 'macro', 'weighted', 'samples']:
                r += (1.0 - f1_score(overAllGroundTruth, item_arr > baseLine, average=am))
            return r

        def entropyLoss(item):
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            entropy_arr = -np.log2(item_arr) * item_arr - np.log2(1 - item_arr) * (1 - item_arr)
            overall_entropy = -np.log2(overAllGroundTruth) * overAllGroundTruth - np.log2(1 - overAllGroundTruth) * (1 - overAllGroundTruth)
            overall_entropy = np.mean(overall_entropy)
            each_entropy = np.mean(entropy_arr)
            return overall_entropy - each_entropy

        tm = time.time()
        print("FEL RKL ETL combine:",combine_method,end="")

        model = torch.load(model_path)
        model.train(True)  # 保持 dropout 开启

        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

        # 防止死循环
        acquire_document_num = acquire_document_num if acquire_document_num <= len(new_datapoints) else len(
            new_datapoints)

        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []

        for iter_batch, data in enumerate(data_batches):
            print('\rCombine acquire batch {}/{}'.format(iter_batch, len(data_batches)), end='')

            batch_data_numpy = data['data_numpy']
            batch_data_points = data['data_points']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())

            score_arr = []

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    # output = model(words_q, words_a, wordslen_q, wordslen_a)
                    pass
                elif model_name == 'CNN':
                    output = model(X)

                score = torch.sigmoid(output).data.cpu().numpy().tolist()
                # score = torch.softmax(output,dim=1).data.cpu().numpy().tolist() # 测试softmax
                # score = F.logsigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))

            new_score_seq = []  # size: btach_size * nsample * nlabel
            for m in range(len(Y)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))

            for index, item in enumerate(new_score_seq):
                # new_xxx shape: batch_size * nsample * nlabel
                # item    shape: nsample * nlabel
                obj = {}
                obj["id"] = pt
                obj["elF1"] = F1Loss(item)
                obj["elRK"] = rankingLoss4(item)
                obj["elET"] = entropyLoss(item)

                _delt_arr.append(obj)
                pt += 1

# ------------------ combine method -------------------- #
        _delt_arr_F1 = sorted(_delt_arr, key=lambda o: o["elF1"], reverse=True)  # 从大到小排序
        _delt_arr_RK = sorted(_delt_arr, key=lambda o: o["elRK"], reverse=True)  # 从大到小排序
        _delt_arr_ET = sorted(_delt_arr, key=lambda o: o["elET"], reverse=True)  # 从大到小排序
        cur_indices = set()

        if combine_method == "FERKETL":
            for i in range(acquire_document_num):
                cur_indices.add(new_datapoints[_delt_arr_F1[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_RK[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_ET[i]["id"]])
            print(" reduency:",len(cur_indices)-acquire_document_num,end=' ')
            cur_indices = self.get_submodular(dataset,cur_indices,acquire_document_num,
                                                  model_path=model_path,model_name=model_name,returned=True)
        elif combine_method == "FERKL":
            for i in range(acquire_document_num):
                cur_indices.add(new_datapoints[_delt_arr_F1[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_RK[i]["id"]])
            print(" reduency:", len(cur_indices) - acquire_document_num, end=' ')
            cur_indices = self.get_submodular(dataset, cur_indices, acquire_document_num,
                                              model_path=model_path, model_name=model_name, returned=True)
        else:
            assert False # not programned

        print("",time.time()-tm,'s')

        self.savedData.append({"added_index": cur_indices,
                               "index2id": {_index: p[3] for _index, p in enumerate(new_dataset)},
                               "_delt_arr": _delt_arr})

        if not returned:
            self.update_train_index(cur_indices)
        else:
            return None,cur_indices


    def get_FELplusRKL(self, dataset, model_path, acquire_document_num,
                nsamp=100, model_name='', returned=False, thisround=-1,):
        # 其实函数内部是乘法的，这里误写成加法了
        def rankingLoss4(item):
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # each RL3
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            each_rl = np.mean((np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1)))

            # overall RL3
            sorted_item_arr = item_arr[:, overAllGroundTruth.argsort()]
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            overall_rl = np.mean(np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1))
            return each_rl - overall_rl


        def F1Loss(item):
            from sklearn.metrics import f1_score
            baseLine = 0.5
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0) > baseLine

            overAllGroundTruth = np.tile(overAllGroundTruth, (item_arr.shape[0], 1))
            r = 0
            for am in ['micro', 'macro', 'weighted', 'samples']:
                r += (1.0 - f1_score(overAllGroundTruth, item_arr > baseLine, average=am))
            return r

        model = torch.load(model_path)
        model.train(True)  # 保持 dropout 开启

        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

        # 防止死循环
        acquire_document_num = acquire_document_num if acquire_document_num <= len(new_datapoints) else len(
            new_datapoints)

        print('FEL: preparing batch data', end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []

        for iter_batch, data in enumerate(data_batches):
            print('\rFEL acquire batch {}/{}'.format(iter_batch, len(data_batches)), end='')

            batch_data_numpy = data['data_numpy']
            batch_data_points = data['data_points']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())

            score_arr = []

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    # output = model(words_q, words_a, wordslen_q, wordslen_a)
                    pass
                elif model_name == 'CNN':
                    output = model(X)

                score = torch.sigmoid(output).data.cpu().numpy().tolist()
                # score = torch.softmax(output,dim=1).data.cpu().numpy().tolist() # 测试softmax
                # score = F.logsigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))

            new_score_seq = []  # size: btach_size * nsample * nlabel
            for m in range(len(Y)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))

            for index, item in enumerate(new_score_seq):
                # new_xxx shape: batch_size * nsample * nlabel
                # item    shape: nsample * nlabel
                obj = {}
                obj["id"] = pt
                obj["el"] = F1Loss(item) * rankingLoss4(item)

                if obj["el"] < -1e-10:
                    print("elo error:", obj["el"])
                    exit()

                _delt_arr.append(obj)
                pt += 1
        print()

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True)  # 从大到小排序

        cur_indices = set()
        i = 0

        while len(cur_indices) < acquire_document_num:
            try:
                cur_indices.add(new_datapoints[_delt_arr[i]["id"]])
                i += 1
            except:
                print(acquire_document_num)
                print(i)
                print(type(new_datapoints), len(new_datapoints))
                print(new_datapoints[:10])
                print(_delt_arr[i])
                assert False

        if not returned:
            self.update_train_index(cur_indices)
        else:
            dataset_pool = []

            return dataset_pool, cur_indices

        self.savedData.append({"added_index": cur_indices,
                               "index2id": {_index: p[3] for _index, p in enumerate(new_dataset)},
                               "_delt_arr": _delt_arr})

    def get_submodular_then_EL(self, dataset, model_path, acquire_document_num,
               nsamp=100, model_name='', thisround=-1,):

        submodular_k = 3
        tm = time.time()

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

        # 防止死循环
        candidate_num = acquire_document_num*submodular_k \
            if acquire_document_num*submodular_k <= len(new_datapoints) \
            else len(new_datapoints)

        candidate_set = self.get_submodular(dataset, new_datapoints, candidate_num , model_path=model_path,
                            model_name=model_name,returned=True)
        candidate_set = sorted(list(candidate_set))
        #------------------------------- submodular step end ----------------------------------#
        def rankingLoss4(item):
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # each RL3
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            each_rl = np.mean((np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1)))

            # overall RL3
            sorted_item_arr = item_arr[:, overAllGroundTruth.argsort()]
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            overall_rl = np.mean(np.mean(positive_item_arr, axis=1) - np.mean(negitive_item_arr, axis=1))
            return each_rl - overall_rl

        rkl=rankingLoss4
        model = torch.load(model_path)
        model.train(True)  # 保持 dropout 开启

        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j in candidate_set]

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j in candidate_set]

        # 防止死循环
        acquire_document_num = acquire_document_num if acquire_document_num <= len(new_datapoints) else len(new_datapoints)

        # print('RKL: preparing batch data', end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []

        for iter_batch, data in enumerate(data_batches):
            # print('\rRKL acquire batch {}/{}'.format(iter_batch, len(data_batches)), end='')

            batch_data_numpy = data['data_numpy']
            batch_data_points = data['data_points']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())

            score_arr = []

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    # output = model(words_q, words_a, wordslen_q, wordslen_a)
                    pass
                elif model_name == 'CNN':
                    output = model(X)

                score = torch.sigmoid(output).data.cpu().numpy().tolist()
                # score = torch.softmax(output,dim=1).data.cpu().numpy().tolist() # 测试softmax
                # score = F.logsigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))

            new_score_seq = []  # size: btach_size * nsample * nlabel
            for m in range(len(Y)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))

            for index, item in enumerate(new_score_seq):
                # new_xxx shape: batch_size * nsample * nlabel
                # item    shape: nsample * nlabel
                obj = {}
                obj["id"] = pt
                obj["el"] = rkl(item)

                if obj["el"] < -1e-10:
                    print("elo error:", obj["el"])
                    exit()

                _delt_arr.append(obj)
                pt += 1

        print()

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True)  # 从大到小排序

        cur_indices = set()
        i = 0

        while len(cur_indices) < acquire_document_num:
            try:
                cur_indices.add(new_datapoints[_delt_arr[i]["id"]])
                i += 1
            except:
                print(acquire_document_num)
                print(i)
                print(type(new_datapoints), len(new_datapoints))
                print(new_datapoints[:10])
                print(_delt_arr[i])
                assert False


        self.update_train_index(cur_indices)
        print('STR time consuming： %d seconds:' % (time.time() - tm))



        pass


    def get_dete(self, dataset, model_path, acquire_document_num,
               model_name='', dete_method = "SIM", returned=False, thisround=-1,):

    # SIM
    # ETY
        tm = time.time()
        if dete_method == "SIM": # submodular
            # id that not in train_index
            new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

            self.get_submodular(dataset, new_datapoints, acquire_document_num, model_path=model_path,
                                 model_name=model_name)
        else:
            model = torch.load(model_path)
            model.train(False)  # 保持 dropout 关闭

            # data without id
            new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]

            # id that not in train_index
            new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

            print('dete : preparing batch data', end='')
            data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

            score_arr = [] # 获取模型对未标记样本的输出
            for iter_batch, data in enumerate(data_batches):
                print('\rdete acquire batch {}/{}'.format(iter_batch, len(data_batches)), end='')

                batch_data_numpy = data['data_numpy']
                batch_data_points = data['data_points']

                X = batch_data_numpy[0]
                Y = batch_data_numpy[1]

                if self.usecuda:
                    X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
                else:
                    X = Variable(torch.from_numpy(X).long())

                if model_name == 'CNN':
                    output = model(X)
                else:
                    assert False

                score = torch.sigmoid(output).data.cpu().numpy().tolist()

                score_arr.extend(score)
            assert len(score_arr) == len(new_dataset)

#-----------------------------------------------------------------------#
            if dete_method == "ETY": # entropy
                item_arr = np.array(score_arr)
                entropy_arr = -np.log2(item_arr) * item_arr - np.log2(1 - item_arr) * (1 - item_arr)
                entropy_arr = np.sum(entropy_arr,axis=1)
                arg = np.argsort(entropy_arr)[-acquire_document_num:] # entropy最大的几个样本的id
                cur_indices = set()
                for i in arg:
                    cur_indices.add(new_datapoints[i])
                    self.update_train_index(cur_indices)
            else:
                assert False #"Not Programmed"

        print('dete time consuming： %d seconds:' % (time.time() - tm))



    def get_submodular(self, data,unlabel_index, acquire_questions_num, model_path='', model_name='', returned=False):
        def greedy_k_center(labeled, unlabeled, amount):
        # input:
        ## labeled features
        ## unlabeled features
        ## amount to be chosen in unlabeled
            greedy_indices = []
            # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
            min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            for j in range(1, labeled.shape[0], 100):
                if j + 100 < labeled.shape[0]:
                    dist = distance_matrix(labeled[j:j + 100, :], unlabeled)
                else:
                    dist = distance_matrix(labeled[j:, :], unlabeled)
                min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
                min_dist = np.min(min_dist, axis=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))

            # iteratively insert the farthest index and recalculate the minimum distances:
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)
            for i in range(amount - 1):
                dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
                min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
                min_dist = np.min(min_dist, axis=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))
                farthest = np.argmax(min_dist)
                greedy_indices.append(farthest)

            return np.array(greedy_indices)

        sample_feature = self.getSimilarityMatrix(data, model_path, model_name, feature_only=True)
        unlabel = list(unlabel_index)
        labeled = list(self.train_index)
        labeled_feature = sample_feature[labeled]
        unlabel_feature = sample_feature[unlabel]
        sel_indices = greedy_k_center(labeled_feature, unlabel_feature, acquire_questions_num)
        cur_indices = np.array(unlabel)[sel_indices].tolist()
        if returned:
            return cur_indices
        else:
            self.update_train_index(cur_indices)

    def getSimilarityMatrix(self, dataset, model_path='', model_name='', batch_size=800, feature_only=False):
        '''
        :param feature_only: 表示返回特征还是相似度矩阵
        '''

        model = torch.load(model_path)
        model.train(False)

        # 对剩余样本池创建batch
        data_batches = create_batches(dataset, batch_size=batch_size, order='no')

        temp_feature = []
        for iter_batch,data in enumerate(data_batches):
            batch_data_numpy  = data['data_numpy']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())

            if model_name == 'BiLSTM':
                # output = model(words_q, words_a, wordslen_q, wordslen_a)
                pass
            elif model_name == 'CNN':
                # 2020 08 12 修改为 features_with_pred
                # output = model.features(X)
                output = model.features_with_pred(X)
            temp_feature.extend(output.data.cpu().numpy().tolist())

        features = np.stack(temp_feature, axis=0)

        if feature_only:
            return features

        similarity = cosine_similarity(features) + 1
        return similarity

    def obtain_data(self, data, model_path=None, model_name=None, acquire_num=2,
                    method='random', sub_method='', unsupervised_method='', round = 0):

        print("sampling method：" + sub_method)

        if model_path == "":
            print("First round of sampling")
            self.get_random(data, acquire_num)
        else:
            if method == 'random':
                self.get_random(data, acquire_num)
            elif method == 'dete':
                self.get_dete(data,model_path,acquire_num,model_name,dete_method=sub_method,thisround=round)
            elif method == 'no-dete': # Bayesian neural network based method
                if sub_method == 'DAL':
                    # 普通DAL
                    self.get_DAL(data, model_path, acquire_num, model_name=model_name)

                    # smDAL
                    # _,unlabeled_index = self.get_DAL(data, model_path, acquire_num*2, model_name=model_name,returned=True)
                    # self.get_submodular(data,unlabeled_index,acquire_num,model_path=model_path,model_name=model_name)

                    # dsmDAL dynamic submodular DAL
                    # _,unlabeled_index = self.get_DAL(data, model_path,
                    #                                  int(acquire_num*( max(2,6-0.5*round) )) ,
                    #                                  model_name=model_name,returned=True)
                    # self.get_submodular(data,unlabeled_index,acquire_num,model_path=model_path,model_name=model_name)
                elif sub_method == 'RS2HEL': # random sampling to instance with high el values
                    self.get_RS2HEL(data, model_path, acquire_num, model_name=model_name,thisround=round)
                elif sub_method == "BALD":
                    self.get_RKL(data, model_path, acquire_num, rklNo='bald', model_name=model_name, thisround=round)
                elif sub_method == "RKL":
                    # # # 普通RKL
                    self.get_RKL(data, model_path, acquire_num, model_name=model_name,thisround=round)

                    # dsm RKL
                    # _, unlabeled_index = self.get_RKL(data, model_path,
                    #                                   int(acquire_num * (max(2.5, 6 - 0.5 * round))),
                    #                                   model_name=model_name, returned=True)
                    # self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                    #                     model_name=model_name)
                elif sub_method == "RKLfp":
                    self.get_RKL(data, model_path, acquire_num, rklNo='4fp', model_name=model_name, thisround=round)

                elif sub_method == "RKL6":
                    #233
                    self.get_RKL(data, model_path, acquire_num, rklNo=6, model_name=model_name, thisround=round)
                elif sub_method == "RKL7":
                    #233
                    self.get_RKL(data, model_path, acquire_num, rklNo=7, model_name=model_name, thisround=round)
                elif sub_method == "RKL9":
                    # 233
                    self.get_RKL(data, model_path, acquire_num, rklNo=9, model_name=model_name, thisround=round)

                elif sub_method == "DRL":
                    # 考虑 点密度 的 RKL, 看做一种排除异常值的方法？
                    # el = rkl * density, 密度低的点，对模型提升的贡献不如密度高的大
                    self.get_RKL(data, model_path, acquire_num, model_name=model_name,density=True,thisround=round)
                elif sub_method == "smDRL":
                    _, unlabeled_index = self.get_RKL(data, model_path, 2*acquire_num,
                                                      model_name=model_name, density=True,
                                                      thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                                        model_name=model_name)

                elif sub_method == "dsm1RKL4":
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num*2, model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path, model_name=model_name)
                elif sub_method == "dsm2RKL4":
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num*4, model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path, model_name=model_name)
                elif sub_method == "dsm3RKL4":
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num*max(1.0,(12-1.5*round)), model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path, model_name=model_name)
                elif sub_method == "dsm4RKL4":
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num*max(1.0,(3-0.2*round)), model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path, model_name=model_name)
                elif sub_method == "dsm5RKL4":
                    temp = [2,2,2,2,2,2,2,2,2,2, 1.5,1.5,1.5,1.5,1.5, 1,1,1,1,1,1,1,1,1,1 ]
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num*temp[round], model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path, model_name=model_name)
                elif sub_method == "dsm6RKL4":
                    temp = [2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1]
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num * temp[round],
                                                      model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                                        model_name=model_name)
                elif sub_method == "dsm7RKL4":
                    temp = [2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4,  2, 2, 2, 2, 1.5, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1]
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num * temp[round],
                                                      model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                                        model_name=model_name)
                elif sub_method == "dsm8RKL4":
                    temp = [2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4,4,4,4 , 4,4,4,4, 4 ]
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num * temp[round],
                                                      model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                                        model_name=model_name)
                elif sub_method == "dsm9RKL4":
                    _, unlabeled_index = self.get_RKL(data, model_path, acquire_num * max((0.25*round),1),
                                                      model_name=model_name, thisround=round, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                                        model_name=model_name)




                elif sub_method == "STR":
                    self.get_submodular_then_EL(data,model_path,acquire_num,model_name=model_name,thisround=round)
                    #

                elif sub_method == "MML":
                    self.get_RKL(data, model_path, acquire_num, model_name=model_name, rklNo='mml', thisround=round)
                    #

                # 信息熵 Loss
                elif sub_method == "ETL":
                    # 信息熵 Loss
                    self.get_RKL(data, model_path, acquire_num, model_name=model_name, rklNo='et', thisround=round)
                elif sub_method == "EEL":
                    # 认为信息熵越大 且 loss 越大 的 越好
                    # 使用overall_entropy * entropy_EL
                    self.get_RKL(data, model_path, acquire_num, model_name=model_name, rklNo='ee', thisround=round)


                elif sub_method == "FEL":
                    self.get_FEL(data, model_path, acquire_num, model_name=model_name,thisround=round)
                    # 233
                elif sub_method == "dsm2RKL":
                    _, unlabeled_index = self.get_RKL(data, model_path,
                                                      int(acquire_num * (max(1.1, 4 - round))),
                                                      model_name=model_name, returned=True)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                                        model_name=model_name)
                elif sub_method == "DARKL":
                    _, DAL_unlabeled_index = self.get_DAL(data, model_path,
                                                      acquire_num,
                                                      model_name=model_name, returned=True)
                    _, RKL_unlabeled_index = self.get_RKL(data, model_path,
                                                      acquire_num,
                                                      model_name=model_name, returned=True)
                    unlabeled_index = set()
                    unlabeled_index.update(DAL_unlabeled_index)
                    unlabeled_index.update(RKL_unlabeled_index)
                    self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                                        model_name=model_name)
                    print("DARK {} redundancy ...".format(len(unlabeled_index)-acquire_num))
                elif sub_method == "FERKL":
                    # _, FEL_unlabeled_index = self.get_FEL(data, model_path,
                    #                                       acquire_num,
                    #                                       model_name=model_name, returned=True)
                    # _, RKL_unlabeled_index = self.get_RKL(data, model_path,
                    #                                       acquire_num,
                    #                                       model_name=model_name, returned=True)
                    # unlabeled_index = set()
                    # unlabeled_index.update(FEL_unlabeled_index)
                    # unlabeled_index.update(RKL_unlabeled_index)
                    # self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                    #                     model_name=model_name)
                    # print("FERK {} redundancy ...".format(len(unlabeled_index) - acquire_num))
                    self.get_FEL_RKL_ETL(data, model_path, acquire_num, model_name=model_name, combine_method="FERKL")
                elif sub_method == "FERKETL":
                    self.get_FEL_RKL_ETL(data, model_path,acquire_num,model_name=model_name,combine_method="FERKETL")
                    #
                elif sub_method == "FELplusRKL":
                    # # # 普通
                    self.get_FELplusRKL(data, model_path, acquire_num, model_name=model_name, thisround=round)
                else:
                    assert 'not progressed'
            else:
                raise NotImplementedError()

        return 1




class Acquisition_old(object):

    def __init__(self, train_data,
                 seed=0,
                 usecuda=True,
                 answer_count=5,
                 cuda_device=0,
                 batch_size=1000,
                 cal_Aleatoric_uncertainty=False,
                 submodular_k=4):
        self.answer_count = answer_count  #answer number for each question
        self.questions_num = int(len(train_data) / answer_count) #question number in the dataset
        self.train_index = set()  #the index of the labeled samples
        self.pseudo_train_data = []
        self.npr = np.random.RandomState(seed)
        self.usecuda = usecuda
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.cal_Aleatoric_uncertainty = cal_Aleatoric_uncertainty
        self.submodular_k = submodular_k

    #-------------------------Random sampling-----------------------------
    def get_random(self, dataset, acquire_questions_num, returned=False):

            question_indices = [self.answer_count * x for x in range(self.questions_num)]
            random_indices = self.npr.permutation(self.questions_num)
            random_question_indices = [question_indices[x] for x in random_indices]

            cur_indices = set()
            sample_q_indices = set()
            i = 0
            while len(cur_indices) < acquire_questions_num * self.answer_count:
                if random_question_indices[i] not in self.train_index:
                    sample_q_indices.add(random_question_indices[i])
                    for k in range(self.answer_count):
                        cur_indices.add(random_question_indices[i] + k)
                i += 1
            if not returned:
                self.train_index.update(cur_indices)
            else:
                return sample_q_indices

    #--------------------------some related active learning methods: var，margin，entropy，me-em，lc
    def get_sampling(self, dataset, model_path, acquire_questions_num,
                     nsamp=100,
                     model_name='',
                     quota='me-em',
                     _reverse=False,
                     deterministic=False, #whether adopt bayesian neural network
                     returned=False
                    ):

        if quota == 'me-em' or quota == 'entropy'\
                or quota == 'mstd' or\
                quota == 'mstd-unregluar' or quota == 'mstd-regluar':
            _reverse = True

        model = torch.load(model_path)
        if deterministic:
            model.train(False)
            nsamp = 1
        else:
            model.train(True)
        tm = time.time()

        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]
        new_question_points = [new_datapoints[x * self.answer_count] for x in range(int(len(new_datapoints)/self.answer_count))]

        print("sample remaining in the pool:%d" % len(new_datapoints))

        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []
        for data in data_batches:

            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda(self.cuda_device)
                words_a = Variable(torch.LongTensor(words_a)).cuda(self.cuda_device)
            else:
                words_q = Variable(torch.LongTensor(words_q))
                words_a = Variable(torch.LongTensor(words_a))

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            ###
            sort_info = data['sort_info']

            tag_arr = []
            score_arr = []
            real_tag_arr = []

            for itr in range(nsamp):
                if model_name == 'BiLSTM':
                    output = model(words_q, words_a, wordslen_q, wordslen_a)
                elif model_name == 'CNN':
                    output = model(words_q, words_a)
                output = F.softmax(output, dim=1)
                score = torch.max(output, dim=1)[0].data.cpu().numpy().tolist()
                tag = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()

                st = sorted(zip(sort_info, score, tag, data['tags']),key=lambda p: p[0])
                _, origin_score, origin_tag, real_tag = zip(*st)

                tag_arr.append(list(origin_tag))
                score_arr.append(list(origin_score))
                real_tag_arr.append(list(real_tag))

            for i in range(len(score_arr)):
                for j in range(len(score_arr[i])):
                    if int(tag_arr[i][j]) == 0:
                        score_arr[i][j] = 1 - score_arr[i][j]

            new_score_seq = []
            for m in range(len(words_q)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            entropy_mean = []
            for i in range(int(len(new_score_seq) / self.answer_count)):
                all = 0
                for j in range(nsamp):
                    temp = []
                    for k in range(self.answer_count):
                        temp.append(new_score_seq[i * self.answer_count + k][j])
                    temp = np.array(temp)
                    temp = (temp + 1e-8) / np.sum(temp + 1e-8)
                    em = -np.sum((temp + 1e-8) * np.log2(temp + 1e-8))
                    all += em

                entropy_mean.append(float(float(all)/float(nsamp)))

            var_mean = []
            if quota == 'mstd-unregluar':
                for i in range(int(len(new_score_seq) / self.answer_count)):
                    all = 0
                    for j in range(self.answer_count):
                        temp = []
                        for k in range(nsamp):
                            temp.append(new_score_seq[i * self.answer_count + j][k])
                        temp = np.array(temp)
                        # temp = (temp + 1e-8) / np.sum(temp + 1e-8)
                        all += temp.std()
                    var_mean.append(float(float(all) / float(self.answer_count)))
            elif quota == 'mstd-regluar':
                for i in range(len(new_score_seq) // self.answer_count):
                    all = []
                    for j in range(nsamp):
                        temp = []
                        for k in range(self.answer_count):
                            temp.append(new_score_seq[i * self.answer_count + k][j])
                        temp = np.array(temp)
                        temp = (temp + 1e-8) / np.sum(temp + 1e-8)
                        all.append(temp)
                    all = np.array(all).T
                    assert all.shape == (self.answer_count, 100)
                    var_mean.append(np.mean(np.std(all, axis=1)))

            mean_score = []
            for i in range(len(new_score_seq) // self.answer_count):
                all = np.zeros(shape=(self.answer_count))
                for j in range(nsamp):
                    temp = []
                    for k in range(self.answer_count):
                        temp.append(new_score_seq[i * self.answer_count + k][j])
                    temp = np.array(temp)
                    temp = (temp + 1e-8) / np.sum(temp + 1e-8)
                    all += temp
                all /= nsamp
                mean_score.extend(all.tolist())

            un_regular_mean_score = []
            for arr in new_score_seq:
                all = 0
                for sc in arr:
                    all += sc
                un_regular_mean_score.append(float(float(all) / float(nsamp)))

            cutted_mean_score = []
            for i in range(int(len(mean_score)/self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(mean_score[i * self.answer_count + j])
                temp = sorted(temp, reverse = True)
                cutted_mean_score.append(temp)

            un_regular_cutted_mean_score = []
            for i in range(int(len(un_regular_mean_score) / self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(un_regular_mean_score[i * self.answer_count + j])
                un_regular_cutted_mean_score.append(temp)

            for i in range(len(cutted_mean_score)):
                item = cutted_mean_score[i]
                if quota == 'var':
                    _delt = np.array(item).var()
                elif quota == 'margin':
                    _delt = item[0] - item[1]
                elif quota == 'lc':
                    _delt = item[0]
                elif quota == "entropy":
                    item = np.array(item)
                    item = (item + 1e-8) / np.sum(item + 1e-8)
                    _delt = -np.sum((item + 1e-8) * np.log2(item + 1e-8))
                elif quota == "me-em":
                    item = np.array(item)
                    item = (item + 1e-8) / np.sum(item + 1e-8)
                    me = -np.sum((item + 1e-8) * np.log2(item + 1e-8))
                    _delt = me - entropy_mean[i]

                obj = {}
                obj["q_id"] = pt
                obj["real_id"] = new_question_points[pt]
                obj["delt"] = _delt
                obj["origin_score"] = un_regular_cutted_mean_score[i]
                _delt_arr.append(obj)

                pt += 1

        _delt_arr = sorted(_delt_arr, key=lambda o: o["delt"], reverse=_reverse)

        cur_indices = set()
        sample_q_indices = set()
        i = 0
        while len(cur_indices) < acquire_questions_num * self.answer_count:
            sample_q_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            for k in range(self.answer_count):
                cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + k)
            i += 1
        if  not returned:
            self.train_index.update(cur_indices)
            print ('time consuming： %d seconds:' % (time.time() - tm))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, sample_q_indices

    def get_BALD(self, dataset, model_path,
                 acquire_questions_num,
                 nsamp=100,
                 model_name='',
                 top_1=True,
                 returned=False
                 ):

        model = torch.load(model_path)
        model.train(True)
        tm = time.time()

        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]
        new_question_points = [new_datapoints[x * self.answer_count] for x in
                               range(int(len(new_datapoints) / self.answer_count))]

        print("sample remaining in the pool:%d" % len(new_datapoints))

        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []
        for data in data_batches:

            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda(self.cuda_device)
                words_a = Variable(torch.LongTensor(words_a)).cuda(self.cuda_device)
            else:
                words_q = Variable(torch.LongTensor(words_q)).cuda(self.cuda_device)
                words_a = Variable(torch.LongTensor(words_a)).cuda(self.cuda_device)

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            ###
            sort_info = data['sort_info']

            tag_arr = []
            score_arr = []
            real_tag_arr = []

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    output = model(words_q, words_a, wordslen_q, wordslen_a)
                elif model_name == 'CNN':
                    output = model(words_q, words_a)
                output = F.softmax(output, dim=1)
                score = torch.max(output, dim=1)[0].data.cpu().numpy().tolist()
                tag = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()

                st = sorted(zip(sort_info, score, tag, data['tags']), key=lambda p: p[0])
                _, origin_score, origin_tag, real_tag = zip(*st)

                tag_arr.append(list(origin_tag))
                score_arr.append(list(origin_score))
                real_tag_arr.append(list(real_tag))

            for i in range(len(score_arr)):
                for j in range(len(score_arr[i])):
                    if int(tag_arr[i][j]) == 0:
                        score_arr[i][j] = 1 - score_arr[i][j]

            new_score_seq = []
            for m in range(len(words_q)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            cutted_score_seq = []
            for i in range(int(len(new_score_seq) / self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(new_score_seq[i * self.answer_count + j])
                cutted_score_seq.append(temp)

            for item in cutted_score_seq:
                tp1 = np.transpose(np.array(item))
                _index = np.argsort(tp1, axis=1).tolist()

                if top_1:  # Only consider the first item in the rank
                    _index = np.argmax(tp1, axis=1)
                    _delt = stats.mode(_index)[1][0]
                else:
                    for i in range(len(_index)):
                        _index[i] = 10000 * _index[i][0] + 1000 * _index[i][1] + 100 * _index[i][2] + 10 * _index[i][
                            3] + _index[i][4]

                    _delt = stats.mode(np.array(_index))[1][0]

                obj = {}
                obj["q_id"] = pt
                obj["delt"] = _delt
                _delt_arr.append(obj)

                pt += 1

        _delt_arr = sorted(_delt_arr, key=lambda o: o["delt"])

        cur_indices = set()
        sample_q_indices = set()
        i = 0

        while len(cur_indices) < acquire_questions_num * self.answer_count:
            sample_q_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + 1)
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + 2)
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + 3)
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + 4)
            i += 1

        if not returned:
            self.train_index.update(cur_indices)
            print('time consuming： %d seconds:' % (time.time() - tm))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, sample_q_indices

    def get_DAL(self, dataset, model_path, acquire_questions_num,
                nsamp=100,
                model_name='',
                returned=False,
                ):

        model = torch.load(model_path)
        model.train(True)
        tm = time.time()

        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]
        new_question_points = [new_datapoints[x * self.answer_count] for x in
                               range(int(len(new_datapoints) / self.answer_count))]

        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []

        for data in data_batches:

            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda(self.cuda_device)
                words_a = Variable(torch.LongTensor(words_a)).cuda(self.cuda_device)
            else:
                words_q = Variable(torch.LongTensor(words_q))
                words_a = Variable(torch.LongTensor(words_a))

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            sort_info = data['sort_info']

            tag_arr = []
            score_arr = []
            real_tag_arr = []
            sigma_total = torch.zeros((nsamp, words_q.size(0)))

            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    output = model(words_q, words_a, wordslen_q, wordslen_a)
                elif model_name == 'CNN':
                    output = model(words_q, words_a)

                score = torch.max(F.softmax(output, dim=1), dim=1)[0].data.cpu().numpy().tolist()
                tag = torch.max(output, dim=1)[1]  #.data.cpu().numpy().tolist()
                tag = tag.data.cpu().numpy().tolist()

                st = sorted(zip(sort_info, score, tag, data['tags']), key=lambda p: p[0])
                _, origin_score, origin_tag, real_tag = zip(*st)

                tag_arr.append(list(origin_tag))
                score_arr.append(list(origin_score))
                real_tag_arr.append(list(real_tag))

            for i in range(len(score_arr)):
                for j in range(len(score_arr[i])):
                    if int(tag_arr[i][j]) == 0:
                        score_arr[i][j] = 1 - score_arr[i][j]

            # new_score_seq = np.array(score_arr).transpose(0, 1).tolist()
            new_score_seq = []
            for m in range(len(words_q)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            cutted_score_seq = []
            for i in range(int(len(new_score_seq) / self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(new_score_seq[i * self.answer_count + j])
                cutted_score_seq.append(temp)

            for index, item in enumerate(cutted_score_seq):   #shape: question_num, 5, nsamp

                def rankedList(rList):
                    rList = np.array(rList)
                    gain = 2 ** rList - 1
                    discounts = np.log2(np.arange(len(rList)) + 2)
                    return np.sum(gain / discounts)

                tp1 = np.transpose(np.array(item)).tolist()
                dList = []
                for i in range(len(tp1)):
                    rL = sorted(tp1[i], reverse=True)
                    dList.append(rankedList(rL))

                # t = np.mean(2 ** np.array(item) - 1, axis=1)
                # rankedt = sorted(t.tolist(), reverse=True)
                # d = rankedList(rankedt)

                item_arr = np.array(item)

                t = np.mean(item_arr, axis=1)
                rankedt = np.transpose(item_arr[(-t).argsort()]).tolist()  #nsamp, 5

                dList2 = []
                for i in range(len(rankedt)):
                    dList2.append(rankedList(rankedt[i]))

                obj = {}
                obj["q_id"] = pt
                obj["el"] = np.mean(np.array(dList)) - np.mean(np.array(dList2))

                if obj["el"] < 0:
                    print("elo error")
                    exit()

                _delt_arr.append(obj)
                pt += 1

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True)

        cur_indices = set()
        sample_q_indices = set()
        i = 0

        while len(cur_indices) < acquire_questions_num * self.answer_count:
            sample_q_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            for k in range(self.answer_count):
                cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + k)

            i += 1

        if not returned:
            print("Active")
            self.train_index.update(cur_indices)
            print('time consuming： %d seconds:' % (time.time() - tm))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, sample_q_indices

    def coreset_sample(self, data, acquire_questions_num, model_path='', model_name='', feature_type='query'):
        def greedy_k_center(labeled, unlabeled, amount):

            greedy_indices = []

            # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
            min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            for j in range(1, labeled.shape[0], 100):
                if j + 100 < labeled.shape[0]:
                    dist = distance_matrix(labeled[j:j + 100, :], unlabeled)
                else:
                    dist = distance_matrix(labeled[j:, :], unlabeled)
                min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
                min_dist = np.min(min_dist, axis=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))

            # iteratively insert the farthest index and recalculate the minimum distances:
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)
            for i in range(amount - 1):
                dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
                min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
                min_dist = np.min(min_dist, axis=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))
                farthest = np.argmax(min_dist)
                greedy_indices.append(farthest)

            return np.array(greedy_indices)

        sample_feature = self.getSimilarityMatrix(data, model_path, model_name, type=feature_type, feature_only=True)
        unlabel = [id for id in range(len(data)) if id not in self.train_index][::self.answer_count]
        unlabel = [id // self.answer_count for id in unlabel]
        labeled = sorted(list(self.train_index))[::self.answer_count]
        labeled = [id // self.answer_count for id in labeled]
        # print('In coreset_sample, labeled size: {}, unlabeled size:{}'.format(len(labeled), len(unlabel)))
        # print('In coreset_sample, sample_feature shape: {}'.format(sample_feature.shape))
        labeled_feature = sample_feature[labeled]
        unlabel_feature = sample_feature[unlabel]
        sel_indices = greedy_k_center(labeled_feature, unlabel_feature, acquire_questions_num)
        sel_indices = np.array(unlabel)[sel_indices].tolist()
        cur_indices = set([ind * self.answer_count + k for ind in sel_indices for k in range(self.answer_count)])
        self.train_index.update(cur_indices)

    def get_submodular(self, uncertainty_sample, data, acquire_questions_num, model_path='', model_name='', feature_type='query'):

        def greedy_k_center(labeled, unlabeled, amount):

            greedy_indices = []
            # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
            min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            for j in range(1, labeled.shape[0], 100):
                if j + 100 < labeled.shape[0]:
                    dist = distance_matrix(labeled[j:j + 100, :], unlabeled)
                else:
                    dist = distance_matrix(labeled[j:, :], unlabeled)
                min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
                min_dist = np.min(min_dist, axis=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))

            # iteratively insert the farthest index and recalculate the minimum distances:
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)
            for i in range(amount - 1):
                dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
                min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
                min_dist = np.min(min_dist, axis=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))
                farthest = np.argmax(min_dist)
                greedy_indices.append(farthest)

            return np.array(greedy_indices)

        sample_feature = self.getSimilarityMatrix(data, model_path, model_name, type=feature_type, feature_only=True)

        labeled = sorted(list(self.train_index))[::self.answer_count]
        labeled = [id // self.answer_count for id in labeled]

        labeled_feature = sample_feature[labeled]
        unlabel_feature = sample_feature[uncertainty_sample]
        # labeled_feature = sample_feature[uncertainty_sample[0:2]]
        # unlabel_feature = sample_feature[uncertainty_sample[2:]]

        sel_indices = greedy_k_center(labeled_feature, unlabel_feature, acquire_questions_num)
        sel_indices = np.array(uncertainty_sample)[sel_indices].tolist()

        # sel_indices = sel_indices + uncertainty_sample[0:2]

        cur_indices = set([ind * self.answer_count + k for ind in sel_indices for k in range(self.answer_count)])
        self.train_index.update(cur_indices)


    def get_submodular2(self, similarity, unlabel, uncertainty_sample, acquire_questions_num):
        _index = ger_submodular(similarity, unlabel, uncertainty_sample,  sel_num=acquire_questions_num)
        cur_indices = set()
        for id in _index:
            for k in range(self.answer_count):
                cur_indices.add(id * self.answer_count + k)
        self.train_index.update(cur_indices)

    def get_submodular3(self, similarity, unlabel, uncertainty_sample, acquire_questions_num):
        _index = ger_submodular_cover(similarity, unlabel, uncertainty_sample,  sel_num = acquire_questions_num)
        cur_indices = set()
        for id in _index:
            for k in range(self.answer_count):
                cur_indices.add(id * self.answer_count + k)
        self.train_index.update(cur_indices)

    # 只考虑问题的动态编码的余弦相似度矩阵
    def getSimilarityMatrix(self, dataset, model_path='', model_name='', batch_size=1000, type='query',
                            feature_only=False):
        '''
        :param feature_only: 表示返回特征还是相似度矩阵
        '''

        model = torch.load(model_path)
        model.train(False)

        # 对剩余样本池创建batch
        data_batches = create_batches(dataset, batch_size=batch_size, order='no')

        temp_q = []
        temp_a = []
        for data in data_batches:
            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda(self.cuda_device)
                words_a = Variable(torch.LongTensor(words_a)).cuda(self.cuda_device)
            else:
                words_q = Variable(torch.LongTensor(words_q))
                words_a = Variable(torch.LongTensor(words_a))

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            if model_name == 'CNN':
                q_f, a_f = model(words_q, words_a, encoder_only=True)
            elif model_name == 'BiLSTM':
                q_f, a_f = model(words_q, words_a, wordslen_q, wordslen_a, encoder_only=True)
            temp_q.extend(list(q_f))
            temp_a.extend(list(a_f))

        q_features = temp_q[::5]
        q_features = np.stack(q_features, axis=0)
        a_features = np.stack(temp_a, axis=0)

        if type == 'query':
            sample_feature = q_features
        if type == 'q-a-concat':
            a_features = a_features.reshape(-1, 5 * a_features.shape[1])
            sample_feature = np.concatenate((q_features, a_features), axis=1)
        elif type == 'q-a-concat-mean':
            q_features = q_features.reshape(q_features.shape[0], 1, q_features.shape[1])
            a_features = a_features.reshape(-1, 5, a_features.shape[1])
            sample_feature = np.mean(np.concatenate((q_features, a_features), axis=1), axis=1)
            assert sample_feature.shape == (q_features.shape[0], q_features.shape[2])
        elif type == 'mean-var':
            a_shape = (a_features.shape[0] // 5, 5, a_features.shape[1])
            a_features = np.reshape(a_features, a_shape)
            mean_feature = np.mean(a_features, axis=1)
            var_feature = np.var(a_features, axis=1)
            sample_feature = np.concatenate((mean_feature, var_feature), axis=1)

        if feature_only:
            return sample_feature

        similarity = cosine_similarity(sample_feature) + 1
        return similarity

    #——————————————————————————————Invoking a sampling strategy to obtain data————————————————————————————————————————————
    def obtain_data(self, data, model_path=None, model_name=None, acquire_questions_num=2,
                    method='random', sub_method='', unsupervised_method='', round = 0):

        print("sampling method：" + sub_method)

        if model_path == "":
            print("First round of sampling")
            self.get_random(data, acquire_questions_num)
        else:
            if unsupervised_method == '':
                if method == 'random':
                    self.get_random(data, acquire_questions_num)
                elif method == 'dete':
                    if sub_method == 'coreset':
                        self.coreset_sample(data, acquire_questions_num, model_path=model_path, model_name=model_name)
                    else:
                        self.get_sampling(data, model_path,
                                          acquire_questions_num, model_name=model_name, quota=sub_method, deterministic=True)
                elif method == 'no-dete':  # Bayesian neural network based method
                    if sub_method == 'BALD':
                        self.get_BALD(data, model_path, acquire_questions_num, model_name=model_name)
                    elif sub_method == 'DAL':
                        self.get_DAL(data, model_path, acquire_questions_num, model_name=model_name)
                    else:
                        self.get_sampling(data, model_path, acquire_questions_num, model_name=model_name, quota=sub_method,
                                          deterministic=False)
                else:
                    raise NotImplementedError()
            elif unsupervised_method == 'submodular':
                temp = []
                for id in range(len(data)):
                    if id not in self.train_index:
                        temp.append(id)

                unlabel = []
                for i in range(len(temp) // self.answer_count):
                    unlabel.append(temp[i * self.answer_count] // self.answer_count)  # 未标注的问题编号

                print(len(unlabel))

                candidate_questions_num = len(unlabel) if len(unlabel) < acquire_questions_num * self.submodular_k \
                    else acquire_questions_num * self.submodular_k
                # candidate_questions_num = len(unlabel) if len(unlabel) < 100 \
                #     else 100

                if method == 'no-dete':
                    if sub_method == 'DAL':
                        dataset_pool, sample_q_indices = self.get_DAL(data, model_path,
                                                                         candidate_questions_num,
                                                                         model_name=model_name, returned=True)

                list(sample_q_indices).sort()
                uncertainty_sample = [id // self.answer_count for id in sample_q_indices]  #采样到的问题编号

                #dynamic_encoder
                # #    type    :    1.query    2.q-a-concat   3.q-a-concat-mean    4.mean-var
                # similarity_matrix = self.getSimilarityMatrix(data, model_path=model_path, model_name=model_name,
                #                                              type='query')
                # self.get_submodular2(similarity_matrix, unlabel, uncertainty_sample, acquire_questions_num)

                self.get_submodular(uncertainty_sample, data, acquire_questions_num,
                                    model_path=model_path, model_name=model_name, feature_type='query')

        return 1


