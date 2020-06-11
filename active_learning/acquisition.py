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


class Acquisition(object):
    def __init__(self, train_data,
                 seed=0,
                 usecuda=True,
                 cuda_device=0,
                 batch_size=1000,
                 cal_Aleatoric_uncertainty=False,
                 submodular_k=4):
        self.train_data = train_data
        self.document_num = len(train_data)
        self.train_index = set()  #the index of the labeled samples
        self.pseudo_train_data = []
        self.npr = np.random.RandomState(seed)
        self.usecuda = usecuda
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.cal_Aleatoric_uncertainty = cal_Aleatoric_uncertainty
        self.submodular_k = submodular_k
        self.savedData = list()
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
            self.train_index.update(sample_indices)
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
            self.train_index.update(cur_indices)
            print('DAL time consuming： %d seconds:' % (time.time() - tm))
            # print('now type1:type2 = {}:{}'.format(
            #                     sum([1 if i<200 else 0 for i in self.train_index ]) ,
            #                     sum([1 if i >= 200 else 0 for i in self.train_index]),
            #       ))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, cur_indices

        self.savedData.append( { "added_index":cur_indices,
                                 "index2id":{_index:p[3] for _index,p in enumerate(new_dataset)},
                                 "_delt_arr":_delt_arr } )

    # el + inconfidence
    def get_DALplusIC(self, dataset, model_path, acquire_document_num,
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

        print('DALplusIC: preparing batch data',end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []

        for iter_batch,data in enumerate(data_batches):
            print('\rDALplusIC acquire batch {}/{}'.format(iter_batch,len(data_batches)),end='')

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
                if thisround==0:
                    obj["el"] = obj["el"] + np.mean( 1-abs(1-2*np.array(item)) )  # 4.2 测试 inconfidence + el
                                                                                # 4.3 前5轮inconfidence,之后都是el
                else:
                    obj["el"] = obj["el"] + \
                                1/np.sqrt(thisround+1)*np.mean( 1-abs(1-2*np.array(item)) )  # 4.4 动态inconfidence权重 1/sqrt(r)*IC

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
            self.train_index.update(cur_indices)
            print('DALplusIC time consuming： %d seconds:' % (time.time() - tm))
            # print('now type1:type2 = {}:{}'.format(
            #                     sum([1 if i<200 else 0 for i in self.train_index ]) ,
            #                     sum([1 if i >= 200 else 0 for i in self.train_index]),
            #       ))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, cur_indices

        self.savedData.append( { "added_index":cur_indices,
                                 "index2id":{_index:p[3] for _index,p in enumerate(new_dataset)},
                                 "_delt_arr":_delt_arr } )

    def get_RS2HEL(self, dataset, model_path, acquire_document_num,
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
        sample_domin = self.npr.permutation(range(sample_domin))

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

        if not returned:
            # print("DAL acquiring:",sorted(cur_indices))
            self.train_index.update(cur_indices)
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

        self.savedData.append( { "added_index":cur_indices,
                                 "index2id":{_index:p[3] for _index,p in enumerate(new_dataset)},
                                 "_delt_arr":_delt_arr } )



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
                assert 'not progressed ...'
            elif method == 'no-dete': # Bayesian neural network based method
                if sub_method == 'DAL':
                    self.get_DAL(data, model_path, acquire_num, model_name=model_name)
                elif sub_method == 'MDAL4.3':
                    if round < 5:
                        self.get_DALplusIC(data, model_path, acquire_num, model_name=model_name)
                    else:
                        self.get_DAL(data, model_path, acquire_num, model_name=model_name)
                    pass
                elif sub_method == 'MDAL4.4':
                    self.get_DALplusIC(data, model_path, acquire_num, model_name=model_name,thisround=round)
                elif sub_method == 'RS2HEL': # random sampling to ins with high el values
                    self.get_RS2HEL(data, model_path, acquire_num, model_name=model_name,)
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


