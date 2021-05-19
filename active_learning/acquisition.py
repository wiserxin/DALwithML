import torch
from torch.autograd import Variable
import numpy as np
import time
import random
from math import floor
from scipy import stats
from scipy.spatial import distance_matrix
from neural.util.utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from .unsupervised import *
from sklearn.cluster import KMeans

class SubMod(object):
    def __init__(self):
        pass

    def getNoDeteMethodWithDAByName(self,methodNick):
        '''
        获取 考虑 DA 数据 的AL方法函数
        :param methodNick:
        :return:
        '''

        # 方差分解分析 for DA data
        def variance_analysis(item, mod="011"):
            '''
            para:
                mod: mod[0]: s 开关
                     mod[1]: s1 开关
                     mod[2]: s2 开关

            func:   对原始样本、生成数据的方差分解分析
            return:
                s1: 认为是多次试验的随机误差，即dropout引起的离差平方和
                    实验中体现为模型的鲁棒性？
                s2: 认为是描述因素各个水平效应的影响，实验中体现为在模型看来
                    data 和 DA data 间的差异程度
            notes:  item 应该是一个 dropout_num * 2 的矩阵，代表了
                    一个label下的两个样本分别采样 dropout_num 次后的结果
            '''
            item_arr = np.array(item)
            res = list()
            if mod[0] == "1":
                s = np.var(item_arr)
                res.append(s)

            if mod[1] == "1":
                # S_e / n / m
                # 认为是多次试验的随机误差，即dropout引起的离差平方和
                s1_arr = np.var(item_arr, axis=0)  # S_e / n
                s1 = np.mean(s1_arr)
                res.append(s1)

            if mod[2] == "1":
                # S_A / n / m
                # 认为是描述因素各个水平效应的影响，即两个样本分别预测结果的差异程度
                overAllGroundTruth = np.mean(item_arr, axis=0)
                s2_arr = np.var(overAllGroundTruth)
                s2 = s2_arr  # S_A / n / m
                res.append(s2)

            # s3_arr = np.var(item_arr,axis=1) # 统计书上没有特殊的解释，直观上是一个样本内部多个标签之间的差异程度。
            # 其长度为dropout次数

            #     print(s1_arr)
            #     print(s1)
            #     print(s2_arr)
            #     print(s2)
            #     print(s,"=",s1,"+",s2)

            return res

        def posLabel(overAllGroundTruth):
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            return list(np.argsort(overAllGroundTruth)[-positive_num:])

        def getPredictLabels(item_arr):
            # get scoers and Pos Labels for a no-dete item arr
            predictScore_arr = np.array(item_arr)
            predictScore = np.mean(predictScore_arr, axis=0)
            predictLabels = posLabel(predictScore)
            return predictScore, predictLabels

        def varRatiosAndStandardDeviation(item0,item1, mod="010"):
            '''

            :param item0:
            :param item1:
            :param mod:
                "100": using s
                "010": using s1
                "001": using s2
                "011": ~
                ......
            :return:
            '''

            a_ori = item0
            a_gen = item1
            _, l_ori = getPredictLabels(item0)
            _, l_gen = getPredictLabels(item1)

            # pos label of all: both ori and gen
            l_all = list((set(l_ori).union(set(l_gen))))
            # pos label of all: both ori and gen
            l_les = list(set(l_ori) - (set(l_ori) - set(l_gen)))

            # arr of l_all columns in ori
            a_la_ori = a_ori[:, l_all]
            # arr of l_all columns in gen
            a_la_gen = a_gen[:, l_all]

            a_ll_gen = a_gen[:, l_les]
            a_ll_ori = a_ori[:, l_les]

            # 把原始数据和生成数据的结果结合成一个新矩阵
            score_arr = np.zeros((a_la_gen.shape[1], 2))
            for index, i in enumerate(zip(a_la_gen.T, a_la_ori.T)):
                one_label_arr = np.vstack(i).T

                # s1 and mean
                score_arr[index, :] = [variance_analysis(one_label_arr, mod=mod)[0], np.mean(one_label_arr)]
                pass

            # Max Standard Deviation with generated data
            msd = np.sqrt(np.max(score_arr[:, 0]))
            # Var Ratios with Generated data
            vrg = 1 - np.mean(score_arr[:, 1])
            res = msd + vrg

            return res

        def varRatiosAndStandardDeviationDouble(item0,item1):
            '''

            :param item0:
            :param item1:
            :param mod:
                using s1 and s2
                max(s1,s2)
                ......
            :return:
            '''

            a_ori = item0
            a_gen = item1
            _, l_ori = getPredictLabels(item0)
            _, l_gen = getPredictLabels(item1)

            # pos label of all: both ori and gen
            l_all = list((set(l_ori).union(set(l_gen))))
            # pos label of all: both ori and gen
            l_les = list(set(l_ori) - (set(l_ori) - set(l_gen)))

            # arr of l_all columns in ori
            a_la_ori = a_ori[:, l_all]
            # arr of l_all columns in gen
            a_la_gen = a_gen[:, l_all]

            a_ll_gen = a_gen[:, l_les]
            a_ll_ori = a_ori[:, l_les]

            # 把原始数据和生成数据的结果结合成一个新矩阵
            score_arr = np.zeros((a_la_gen.shape[1], 3))
            for index, i in enumerate(zip(a_la_gen.T, a_la_ori.T)):
                one_label_arr = np.vstack(i).T

                # s1 and mean
                score_arr[index, :2] = variance_analysis(one_label_arr)
                score_arr[index, 2] =  np.mean(one_label_arr)
                pass

            # Max Standard Deviation with generated data
            msd = np.sqrt(np.max(score_arr[:, :2]))
            # Var Ratios with Generated data
            vrg = 1 - np.mean(score_arr[:, 2])
            res = msd + vrg

            return res

        def varRatiosAndStandardDeviationSum(item0,item1):
            return varRatiosAndStandardDeviation(item0,item1,mod="100")

        def varRatios(item0, item1,):
            '''

            :param item0:
            :param item1:
            :return:
            '''

            a_ori = item0
            a_gen = item1
            _, l_ori = getPredictLabels(item0)
            _, l_gen = getPredictLabels(item1)

            # pos label of all: both ori and gen
            l_all = list((set(l_ori).union(set(l_gen))))
            # pos label of all: both ori and gen
            # l_les = list(set(l_ori) - (set(l_ori) - set(l_gen)))

            # arr of l_all columns in ori
            a_la_ori = a_ori[:, l_all]
            # arr of l_all columns in gen
            a_la_gen = a_gen[:, l_all]

            vrg = 1 - np.mean([a_la_ori,a_la_gen])

            return vrg

        def maxStandardDeviation(item0,item1, mod="010"):
            '''

            :param item0:
            :param item1:
            :param mod:
                "100": using s
                "010": using s1
                "001": using s2
                "011": ~
                ......
            :return:
            '''

            a_ori = item0
            a_gen = item1
            _, l_ori = getPredictLabels(item0)
            _, l_gen = getPredictLabels(item1)

            # pos label of all: both ori and gen
            l_all = list((set(l_ori).union(set(l_gen))))
            # pos label of all: both ori and gen
            # l_les = list(set(l_ori) - (set(l_ori) - set(l_gen)))

            # arr of l_all columns in ori
            a_la_ori = a_ori[:, l_all]
            # arr of l_all columns in gen
            a_la_gen = a_gen[:, l_all]

            # a_ll_gen = a_gen[:, l_les]
            # a_ll_ori = a_ori[:, l_les]

            # 把原始数据和生成数据的结果结合成一个新矩阵
            score_arr = np.zeros((a_la_gen.shape[1], 1))
            for index, i in enumerate(zip(a_la_gen.T, a_la_ori.T)):
                one_label_arr = np.vstack(i).T
                # s1
                score_arr[index, :] = [variance_analysis(one_label_arr, mod=mod)[0],]

            # Max Standard Deviation with generated data
            msd = np.sqrt(np.max(score_arr[:, 0]))
            return msd



        methodDic = {'VSD': varRatiosAndStandardDeviation,
                     'VSDD':varRatiosAndStandardDeviationDouble,
                     'VSDS':varRatiosAndStandardDeviationSum,
                     'VRS': varRatios,
                     'MSD': maxStandardDeviation,
                     }
        method = methodDic[methodNick]
        print("Choose method", method, end="\t")
        return method

    def getNoDeteMethodByName(self,methodNick):
        def rankingLoss2(item):  # item = nsamples * labels
            # 使用ground truth 给每个forward pass筛选，计算所有的 neg_label > pos_label 对 的 sum(neg_label-pos_label)
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

        def rankingLoss8(item, mod=0):
            # 思路：
            # 1 仅计算所有的 pos 与 最大的 neg label 间的差值作为Loss？？？
            #   loss =  1 - mean(pos-neg_highest)  值越大越应选中
            #   因为 pos 与 neg 越接近  该样本越不确定
            #   然后 就 el = Loss_each - Loss_overall
            # 2 或者使用 Loss*RKL4  试试 ？就变相的给RKL4加权了
            #   认为 Loss 和 var Ratios 即 vrs 的效果应一致，可以测试一下
            #   即使用 mod 0
            #
            # mod  |  效果
            #  0   |  返回 overall RL
            #  1   |  返回 each  RL
            #  2   |  返回 el  RL = each RL - overall RL      # 这个 el 有缺陷, elo number 接近半数
            #  3   |  返回 RKL4 * overall RL
            #  4   |  返回 overall ( 1-min(pos) )
            #  5   |  返回 overall ( min(pos) - max(neg) )
            #  6   |  返回 each    ( min(pos) - max(neg) )

            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # each RL8
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]
            lowest_positive_item  = positive_item_arr[:,0]
            highest_negitive_item = negitive_item_arr[:,-1]
            each_rl = 1 - np.mean((np.mean(positive_item_arr, axis=1) - highest_negitive_item ))

            # overall RL8
            sorted_overAllGroundTruth = np.sort(overAllGroundTruth)
            positive_item_arr = sorted_overAllGroundTruth[-positive_num:]
            lowest_positive_item  = sorted_overAllGroundTruth[-positive_num]
            highest_negitive_item = sorted_overAllGroundTruth[-positive_num-1]
            overall_rl = 1 - np.mean( np.mean(positive_item_arr) - highest_negitive_item )

            returnList = ["overall_rl", "each_rl", "each_rl-overall_rl", "rankingLoss4(item)*overall_rl", # 0 1 2 3
                          "1-lowest_positive_item", "lowest_positive_item-highest_negitive_item",       # 4 5
                          "np.mean(lowest_positive_item-highest_negitive_item)",                        # 6
                          ]
            return eval(returnList[mod])

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

        def entropyLoss(item,mod=False):
            # ee: entropy*EL ?
            ee = mod
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
            return entropyLoss(item,mod=True)

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

        def varRatios(item):
            # Var Ratios
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            sorted_overAllGroundTruth = sorted(overAllGroundTruth)
            positive_item = sorted_overAllGroundTruth[-positive_num:]
            return 1-np.mean(positive_item) # 1 减去 正标签的出现概率

        def varRatiosLoss(item):
            # EL in var Ratios
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # overall
            sorted_overAllGroundTruth = sorted(overAllGroundTruth)
            overall_ratios = np.mean(sorted_overAllGroundTruth[-positive_num:])

            # each
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            each_ratios = np.mean(positive_item_arr,)

            return each_ratios - overall_ratios

        def label_instance_wise(item,mod=0):
            #  liw : label-instance-wise
            #  0 : lab * ins
            #  1 : label + ins
            #  2 : ins
            #  3 : label

            # 方案 1 ：
            # label wise 仅考虑一个样本的不同forward pass， 此时对于 *pos* label，给出label-wise的计算值 。
            # (因为之前的vrs实验中仅使用pos的效果就足够了)
            # 而instance-wise的值由均值（overAllGroundTruth给出）

            baseline = 0.5
            item_arr = np.array(item)

            overAllGroundTruth = np.mean(item_arr, axis=0)
            # print(overAllGroundTruth)
            positive_num = np.sum(overAllGroundTruth > baseline)
            # print(overAllGroundTruth.size)
            # print(overAllGroundTruth.shape)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1
            # print(overAllGroundTruth)

            overAllGroundTruth_argsort = np.argsort(overAllGroundTruth)  # 按值由小到大 给出 index
            # print(overAllGroundTruth_argsort)

            label_wise_value = overAllGroundTruth[overAllGroundTruth_argsort[-positive_num]] - \
                               overAllGroundTruth[overAllGroundTruth_argsort[-positive_num - 1]]

            pos_args = overAllGroundTruth_argsort[-positive_num:]
            # neg_args = overAllGroundTruth_argsort[:-positive_num]

            # print("-"*30)
            # -------------------------------------------------------------------- #
            # 把一个instance的几个dropout视为不同的instance，并以此为矩阵计算
            # 共有j个label（item的列数），则应计算j个value，我们选用其中 pos_num个
            # instance－wise value
            # forward-wise value
            temp = item_arr.transpose()[pos_args, :]
            item_arr_TF = temp > baseline
            # print(item_arr_TF)

            item_arr_T = temp * item_arr_TF
            # print(item_arr_T)
            item_arr_T += (item_arr_T == 0)  # change all neg into 1 to get the min pos value
            # print(item_arr_T)

            item_arr_F = temp * (item_arr_TF == 0)
            # print(item_arr_F)
            # change all pos into 0 to get the max neg value

            # note that ,
            # if one column has no pos, means that it is very confident to be neg
            # so we let min pos be 1 , to get a higher value of min_pos-max_neg, and less possible to be chosen
            instance_wise_pos_value = np.min(item_arr_T, axis=1) - np.max(item_arr_F, axis=1)
            instance_wise_pos_value = np.mean(instance_wise_pos_value)

            # print(instance_wise_pos_value)

            if mod == 0:
                value = 1-label_wise_value * instance_wise_pos_value
            elif mod == 1:
                value = 2-label_wise_value - instance_wise_pos_value
            elif mod == 2:
                value = 1-instance_wise_pos_value
            elif mod == 3:
                value = 1-label_wise_value
            return value

        def variance_analysis(item):
            '''
            func:   方差分解分析
            return:
                s1: 认为是多次试验的随机误差，即dropout引起的离差平方和
                    实验中体现为模型的鲁棒性？
                s2: 认为是描述因素各个水平效应的影响，实验中体现为
                    pos label 间的差异程度、neg label 间的差异程度
            notes:  这种方法应当设计用在单标签分类问题 即
                    同一个label,不同样本,不同dropout，这样的可解释性更强
                    我在使用中，把label 和样本 对调，由此的可解释性差了一些，
                    而且label数量远小于样本数量，可能导致置信度更差
            '''
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)

            s1_arr = np.var(item_arr, axis=0)  # S_e / n
            s2_arr = np.var(overAllGroundTruth)
            # s3_arr = np.var(item_arr,axis=1) # 统计书上没有特殊的解释，直观上是一个样本内部多个标签之间的差异程度。
            # 其长度为dropout次数

            # S_e / n / m
            # 认为是多次试验的随机误差，即dropout引起的离差平方和
            s1 = np.mean(s1_arr)

            # S_A / n / m
            # 认为是描述因素各个水平效应的影响，即同类label间的差异程度
            s2 = s2_arr  # S_A / n / m

            s = np.var(item_arr)
            # print(s1_arr)
            # print(s1)
            # print(s2_arr)
            # print(s2)
            # print(s,"=",s1,"+",s2)
            return s1, s2

        def meanVarLoss(item, mod=(0.4,0.3,0.3)):
            '''
            input:
                mod : (w_m, w_s1, w_s2)
                w_m : 均值权重
                w_s1: s1权重
                w_s2: s2权重
            return:
                w_m*( (1-(pm-nm))**2 ) + w_s1*( ps1 + ns1 ) + w_s2*(ps2+ns2)
                (1-(pm-nm))**2 : 为了使得方差和均值处在一个数量级，对均值取了平方
                ps1 + ns1      : dropout引起的离差平方和
                ps2+ns2        : label 间的差异程度

            '''

            # 方差越大，该样本模型预测的越不确定
            # 均值差越小，该样本模型预测的越不确定

            # 使用方差分解的话，在el中的这种 求each 处理方式应该是无法解释的。
            # 感觉可以尝试但基本无法拿来写文章
            # 把 item_array 按照各自内部的顺序分成 pos 和 neg 两个 arr
            # item_arr = np.array(item)
            # sorted_item_arr = np.sort(item_arr)
            # positive_item_arr = sorted_item_arr[:,-positive_num:]
            # negitive_item_arr = sorted_item_arr[:,:-positive_num]

            # 把 item_array 按照overAllGroundTruth分成 pos 和 neg 两个 arr
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)

            positive_num = 1 if positive_num == 0 else positive_num
            positive_num = overAllGroundTruth.size - 1 if positive_num == overAllGroundTruth.size else positive_num

            sorted_item_arr = item_arr[:, overAllGroundTruth.argsort()]
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            negitive_item_arr = sorted_item_arr[:, :-positive_num]

            pm = np.mean(positive_item_arr)
            nm = np.mean(negitive_item_arr)

            ps1, ps2 = variance_analysis(positive_item_arr)
            ns1, ns2 = variance_analysis(negitive_item_arr)

            if len(mod) == 3:
                w_m, w_s1, w_s2 = mod
                # 为了使得方差和均值处在一个数量级，对均值取了平方
                return w_m * ((1 - (pm - nm)) ** 2) + w_s1 * (ps1 + ns1) + w_s2 * (ps2 + ns2)
            elif len(mod) == 5:
                w_m, w_s1_p, w_s1_n, w_s2_p, w_s2_n = mod
                return w_m*((1-(pm-nm))**2) + w_s1_p*ps1 + w_s1_n*ns1 + w_s2_p*ps2 + w_s2_n*ns2
            elif len(mod) == 6:
                # an example is (0.4,0, 0.3,0, 0.3,0)
                # each in len==3 splitted into 2 parts: pos and neg
                w_m_p, w_m_n, w_s1_p, w_s1_n, w_s2_p, w_s2_n = mod
                return w_m_p*((1-pm)**2) + w_m_n*((nm-0)**2) + w_s1_p*ps1 + w_s1_n*ns1 + w_s2_p*ps2 + w_s2_n*ns2
            else:
                assert not "defined!"

        def F_variance_analysis(item, mod=0):
            # F 分布 F(k-1,n-k), F值过大拒绝
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)

            positive_num = 1 if positive_num == 0 else positive_num
            positive_num = overAllGroundTruth.size - 1 if positive_num == overAllGroundTruth.size else positive_num

            overall_arg_sort = overAllGroundTruth.argsort()
            positive_columns = overall_arg_sort[-positive_num:]
            negetive_columns = overall_arg_sort[:-positive_num]
            item_arr[:, positive_columns] = 1 - item_arr[:, positive_columns]
            s_dropout, s_labels = variance_analysis(item_arr)

            n,k = np.shape(item)

            if mod == 0:
                return (s_labels/(k-1)) / (s_dropout/(n*k-k))
            else:
                m_pos = np.mean(overAllGroundTruth[positive_columns])
                m_neg = np.mean(overAllGroundTruth[negetive_columns])
                if mod==1:
                    return (s_labels/(k-1)) / (s_dropout/(n*k-k)) * (1 - (m_pos - m_neg))
                elif mod == 2:
                    return (s_labels / (k - 1)) / (s_dropout / (n * k - k)) * (1 - m_pos)


        # 选取策略
        methodDic = {2:rankingLoss2,
                  4:rankingLoss4,
                  5:rankingLoss5,
                  6:rankingLoss6,
                  7:rankingLoss7,
                  8:rankingLoss8,
                  9:rankingLoss9,

                  '4fp':rankingLoss4_first_part,

                  'mml':meanMaxLoss,

                  'et':entropyLoss,
                  'ee':eentropyLoss,
                  'bald': BALD,
                  'vrs' : varRatios,
                  'vrl' : varRatiosLoss,
                  'liw' : label_instance_wise,
                  'mvl' : meanVarLoss,
                  'fva' : F_variance_analysis,
                  }
        method = methodDic[methodNick]
        print("Choose method",method,end="\t")
        return method

    def getDeteMethodByName(self,methodNick):
        def posLabel(overAllGroundTruth):
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1
            return list(np.argsort(overAllGroundTruth)[-positive_num:])

        def predictD(x, y):
            xl = set(posLabel(x))
            yl = set(posLabel(y))
            level = len(xl - yl) + len(yl - xl)
            cl = xl - (xl - yl)  # 共同的label
            # 曼哈顿距离 + level
            return np.sum(np.abs(x[list(cl)] - y[list(cl)])) + level

        methodDic = {
            "pd":predictD,
            "test":123
        }

        method = methodDic[methodNick]

        return method

allSubMethod = SubMod()



class Acquisition(object):
    def __init__(self, train_data,
                 seed=0,
                 usecuda=True,
                 cuda_device=0,
                 batch_size=1000,
                 cal_Aleatoric_uncertainty=False,
                 submodular_k=4,
                 using_generated_data = False,
                 generated_per_sample = 0,
                 generated_used_per_sample = 0,
                 deal_generated_train_index_method="",
                 target_size = 2):
        self.train_data = train_data
        self.generated_train_data = list() # if using_generated_data, input the data later after init
        self.document_num = len(train_data)
        self.train_index = set()  #the index of the labeled samples
        self.generated_train_index = set() # the index of the generated labeled samples
        self.round = -1 # the acquire round :first round 's num is 0; updated by func obtain_data
        self.random_seed = seed
        self.npr = np.random.RandomState(seed)
        self.usecuda = usecuda
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.cal_Aleatoric_uncertainty = cal_Aleatoric_uncertainty
        self.submodular_k = submodular_k

        self.using_generated_data = using_generated_data
        self.generated_per_sample = generated_per_sample
        self.generated_used_per_sample = generated_used_per_sample
        self.deal_generated_train_index_method = deal_generated_train_index_method
        self.deal_generated_train_index_para = None

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

    def update_train_index(self, acquired_set,):
        # 注意调用时应保证 先 self.savedData.append , 再 update_train_index
        # 更新 label_count 和 train_index
        temp = []
        for i in acquired_set:
            temp.append(np.array(self.train_data[i][1].todense()))
        train_Y = np.squeeze(np.stack(temp))
        # print(train_Y.shape)
        each_label_count = np.sum(train_Y, axis=0)
        # print(self.label_count.shape, each_label_count.shape)
        self.label_count = self.label_count + each_label_count
        # print(self.label_count)
        # print(self.label_count.shape)
        # assert False
        self.train_index.update(acquired_set)


        #----------------------------------------------------------------------------------------------------#
        # 更新 self.generated_train_index
        if self.using_generated_data:
            def default():
                acquired_generated_set = {self.generated_per_sample * one_train_index + generated_counter
                                          for one_train_index in acquired_set
                                          for generated_counter in range(0, self.generated_used_per_sample)}
                self.generated_train_index.update(acquired_generated_set)

            def randomDropout():
                default()
                # random dropout selected self.generated_train_index in rate para
                # para :: the rate, between  0-1
                a = list(self.generated_train_index)
                random.shuffle(a)
                a = a[:floor(len(a) * self.deal_generated_train_index_para) // self.batch_size * self.batch_size]  # 确保数据量与batchsize相符，多余的长度会随机丢弃
                self.generated_train_index = set(a)

            def easySlowDown():
                self.generated_used_per_sample = max(0,self.generated_per_sample-self.round//4)
                default()

            deal_genereted_fun = {
                ""    : default,
                "rdr" : randomDropout,
                "esd" : easySlowDown,
            }
            deal_genereted_fun[self.deal_generated_train_index_method]()


        # 更新 saved data , 即每个round选取的train_index 和 generated_data
        if len(self.savedData) == 0:
            self.savedData.append({"train_index":self.train_index, "generated_train_index":self.generated_train_index})
        else:
            self.savedData[-1]["train_index"]=self.train_index
            self.savedData[-1]["generated_train_index"]=self.generated_train_index

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
                rklMod = -1,
                density = False,#开启则挑选更稠密的区间的点
                thisround=-1,):


        rkl = allSubMethod.getNoDeteMethodByName(rklNo)
        # print("RKL",rklNo,end="\t")



        model = torch.load(model_path)
        model.train(True) # 保持 dropout 开启
        tm = time.time()
        if self.using_generated_data:
            using_generated_data = [self.generated_train_data[i*self.generated_per_sample+j]
                                    # for i in range(int(len(self.generated_train_data)/self.generated_per_sample))
                                    for i in range(len(dataset))
                                    for j in range(self.generated_used_per_sample)
                                    ]
            # >> > a = [(i, j) for i in range(5) for j in ('1', '2')]
            # >> > a
            # [(0, '1'), (0, '2'), (1, '1'), (1, '2'), (2, '1'), (2, '2'), (3, '1'), (3, '2'), (4, '1'), (4, '2')]
            # sample_feature = self.getSimilarityMatrixNTimes(dataset, model_path, model_name, feature_only=True)  # 原始数据的特征
            sample_feature_generated, sample_score_generated = self.getSimilarityMatrixNTimes(
                using_generated_data, model_path, model_name, feature_only=True, with_predict=True, nsamp=nsamp)

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
        all_score_arr = []

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

                score = torch.sigmoid(output).data.cpu().numpy()
                # score = torch.sigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))

            # size: btach_size * nsample * nlabel
            # # reshape method 1
            # new_score_seq = []
            # for m in range(len(Y)):
            #     tp = []
            #     for n in range(nsamp):
            #         tp.append(score_arr[n][m].tolist())
            #     new_score_seq.append(tp)
            # # reshape method 2
            # new_score_seq = [ [score_arr[n][m].tolist() for n in range(nsamp)] for m in range(len(Y)) ]
            # # reshape method 3
            new_score_seq = np.stack(score_arr, axis=1)


            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))
            all_score_arr.extend(new_score_seq.tolist())

            for index, item in enumerate(new_score_seq):
                # new_xxx shape: batch_size * nsample * nlabel
                # item    shape: nsample * nlabel
                obj = {}
                obj["id"] = pt
                obj["el"] = rkl(item) if rklMod==-1 else rkl(item,mod=rklMod)

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
                               "item_arr": all_score_arr,
                               "_delt_arr": _delt_arr,
                               "sample_score_generated":sample_score_generated})

        if not returned:
            self.update_train_index(cur_indices)
            print('RKL time consuming： %d seconds:' % (time.time() - tm))
        else:
            dataset_pool = []

            return dataset_pool, cur_indices

    def get_GRKL(self, dataset, model_path, acquire_document_num,
                nsamp=100,
                model_name='',
                returned=False,
                rklNo = 4,      # 选取rkl策略，默认是rkl4
                rklMod = -1,
                density = False,#开启则挑选更稠密的区间的点
                thisround=-1,):


        rkl = allSubMethod.getNoDeteMethodWithDAByName(rklNo)
        # print("RKL",rklNo,end="\t")



        model = torch.load(model_path)
        model.train(True) # 保持 dropout 开启
        tm = time.time()
        if self.using_generated_data:
            using_generated_data = [self.generated_train_data[i*self.generated_per_sample+j]
                                    # for i in range(int(len(self.generated_train_data)/self.generated_per_sample))
                                    for i in range(len(dataset))
                                    for j in range(self.generated_used_per_sample)
                                    ]
            # >> > a = [(i, j) for i in range(5) for j in ('1', '2')]
            # >> > a
            # [(0, '1'), (0, '2'), (1, '1'), (1, '2'), (2, '1'), (2, '2'), (3, '1'), (3, '2'), (4, '1'), (4, '2')]
            # sample_feature = self.getSimilarityMatrixNTimes(dataset, model_path, model_name, feature_only=True)  # 原始数据的特征
            sample_feature_generated, sample_score_generated = self.getSimilarityMatrixNTimes(
                using_generated_data, model_path, model_name, feature_only=True, with_predict=True, nsamp=nsamp)

        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]

        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]

        # 防止死循环
        acquire_document_num = acquire_document_num if acquire_document_num <= len(new_datapoints) else len(new_datapoints)

        print('GRKL: preparing batch data',end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')

        pt = 0
        _delt_arr = []
        elo_count = 0
        all_score_arr = []

        for iter_batch,data in enumerate(data_batches):
            print('\rGRKL acquire batch {}/{} elo num:{}'.format(iter_batch,len(data_batches),elo_count),end='')

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

                score = torch.sigmoid(output).data.cpu().numpy()
                # score = torch.sigmoid(output).data.cpu().numpy().tolist()

                score_arr.append(score)

                # evidence level , using confidence stratgy
                # score_arr.append(torch.abs(score-0.5))

            # size: btach_size * nsample * nlabel
            new_score_seq = np.stack(score_arr, axis=1)


            # print("new_score_seq:",len(new_score_seq),len(new_score_seq[0]),len(new_score_seq[0][0]))
            all_score_arr.extend(new_score_seq.tolist())

            for index, item in enumerate(new_score_seq):
                # new_xxx shape: batch_size * nsample * nlabel
                # item    shape: nsample * nlabel
                obj = {}
                obj["id"] = pt
                obj["el"] = rkl(item,sample_score_generated[new_dataset[index][3]]
                                ) if rklMod==-1 else rkl(item,sample_score_generated[new_dataset[index][3]],mod=rklMod)

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
                               "item_arr": all_score_arr,
                               "_delt_arr": _delt_arr,
                               "sample_score_generated":sample_score_generated})

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

        def F1Loss_fixed(item):
            from sklearn.metrics import f1_score
            baseLine = 0.5
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0) > baseLine

            overAllGroundTruth = np.tile(overAllGroundTruth, (item_arr.shape[0], 1)) # 复制出多行
            r = 0
            for am in ['micro', 'macro']:
                r += (f1_score(overAllGroundTruth, overAllGroundTruth, average=am) -
                      f1_score(overAllGroundTruth, item_arr > baseLine, average=am))
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

        def varRatios(item): # 非EL方法的
            # Var Ratios
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            sorted_overAllGroundTruth = sorted(overAllGroundTruth)
            positive_item = sorted_overAllGroundTruth[-positive_num:]
            return 1-np.mean(positive_item) # 1 减去 正标签的出现概率

        def varRatiosLoss(item):
            # EL in var Ratios
            item_arr = np.array(item)
            overAllGroundTruth = np.mean(item_arr, axis=0)
            positive_num = np.sum(overAllGroundTruth > 0.5)
            if positive_num == 0:
                positive_num = 1
            elif positive_num == overAllGroundTruth.size:
                positive_num = overAllGroundTruth.size - 1

            # overall
            sorted_overAllGroundTruth = sorted(overAllGroundTruth)
            overall_ratios = np.mean(sorted_overAllGroundTruth[-positive_num:])

            # each
            sorted_item_arr = np.sort(item_arr)
            positive_item_arr = sorted_item_arr[:, -positive_num:]
            each_ratios = np.mean(positive_item_arr,)

            return each_ratios - overall_ratios

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

                # get different loss
                if "FE" in combine_method:
                    obj["elF1"] = F1Loss(item)
                if "RKfp" in combine_method:
                    obj["elRKfp"] = rankingLoss4_first_part(item)
                elif "RK" in combine_method:
                    obj["elRK"] = rankingLoss4(item)
                if "ET" in combine_method:
                    obj["elET"] = entropyLoss(item)
                if "VRL" in combine_method:
                    obj["elVR"] = varRatiosLoss(item)
                if "VRS" in combine_method:
                    obj["VRS"] = varRatios(item)

                _delt_arr.append(obj)
                pt += 1

# ------------------ combine method -------------------- #

        cur_indices = set()

        if combine_method == "FERKETL":
            _delt_arr_F1 = sorted(_delt_arr, key=lambda o: o["elF1"], reverse=True)  # 从大到小排序
            _delt_arr_RK = sorted(_delt_arr, key=lambda o: o["elRK"], reverse=True)  # 从大到小排序
            _delt_arr_ET = sorted(_delt_arr, key=lambda o: o["elET"], reverse=True)  # 从大到小排序
            for i in range(acquire_document_num):
                cur_indices.add(new_datapoints[_delt_arr_F1[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_RK[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_ET[i]["id"]])
            print(" reduency:",len(cur_indices)-acquire_document_num,end=' ')
            cur_indices = self.get_submodular(dataset,cur_indices,acquire_document_num,
                                                  model_path=model_path,model_name=model_name,returned=True)
        elif combine_method == "FERKL":
            _delt_arr_F1 = sorted(_delt_arr, key=lambda o: o["elF1"], reverse=True)  # 从大到小排序
            _delt_arr_RK = sorted(_delt_arr, key=lambda o: o["elRK"], reverse=True)  # 从大到小排序
            for i in range(acquire_document_num):
                cur_indices.add(new_datapoints[_delt_arr_F1[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_RK[i]["id"]])
            print(" reduency:", len(cur_indices) - acquire_document_num, end=' ')
            cur_indices = self.get_submodular(dataset, cur_indices, acquire_document_num,
                                              model_path=model_path, model_name=model_name, returned=True)
        elif combine_method == "FERKfpL":
            _delt_arr_F1 = sorted(_delt_arr, key=lambda o: o["elF1"], reverse=True)  # 从大到小排序
            _delt_arr_RKfp = sorted(_delt_arr, key=lambda o: o["elRKfp"], reverse=True)  # 从大到小排序
            for i in range(acquire_document_num):
                cur_indices.add(new_datapoints[_delt_arr_F1[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_RKfp[i]["id"]])
            print(" reduency:", len(cur_indices) - acquire_document_num, end=' ')
            cur_indices = self.get_submodular(dataset, cur_indices, acquire_document_num,
                                              model_path=model_path, model_name=model_name, returned=True)
        elif combine_method == "FEVRL":
            _delt_arr_F1 = sorted(_delt_arr, key=lambda o: o["elF1"], reverse=True)  # 从大到小排序
            _delt_arr_VR = sorted(_delt_arr, key=lambda o: o["elVR"], reverse=True)  # 从大到小排序
            for i in range(acquire_document_num):
                cur_indices.add(new_datapoints[_delt_arr_F1[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_VR[i]["id"]])
            print(" reduency:", len(cur_indices) - acquire_document_num, end=' ')
            cur_indices = self.get_submodular(dataset, cur_indices, acquire_document_num,
                                              model_path=model_path, model_name=model_name, returned=True)
        elif combine_method == "FEVRS":
            _delt_arr_F1 = sorted(_delt_arr, key=lambda o: o["elF1"], reverse=True)  # 从大到小排序
            _delt_arr_VR = sorted(_delt_arr, key=lambda o: o["VRS"], reverse=True)  # 从大到小排序
            for i in range(acquire_document_num):
                cur_indices.add(new_datapoints[_delt_arr_F1[i]["id"]])
                cur_indices.add(new_datapoints[_delt_arr_VR[i]["id"]])
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
        # VRS
        # STD

        def mean_STD(item):
            # Mean-STD
            item_arr = np.array(item)
            item_arr_pow2 = item_arr ** 2
            E1 = np.mean(item_arr_pow2, axis=0)
            E2 = np.mean(item_arr, axis=0) ** 2
            mean_STD = np.mean(np.sqrt(E1 - E2))
            return mean_STD

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

            elif dete_method == "VRS": # var Ratios
                item_arr = np.array(score_arr)
                varRatios_arr = list()
                for overAllGroundTruth in item_arr:
                    positive_num = np.sum(overAllGroundTruth > 0.5)
                    if positive_num == 0:
                        positive_num = 1
                    elif positive_num == overAllGroundTruth.size:
                        positive_num = overAllGroundTruth.size - 1
                    sorted_overAllGroundTruth = sorted(overAllGroundTruth)
                    positive_item = sorted_overAllGroundTruth[-positive_num:]
                    varRatios_arr.append(1-np.mean(positive_item))
                arg = np.argsort(varRatios_arr)[-acquire_document_num:]  # ratios 最大的几个样本的id
                cur_indices = set()
                for i in arg:
                    cur_indices.add(new_datapoints[i])

                self.savedData.append({"added_index": cur_indices,
                                       "index2id": {_index: p[3] for _index, p in enumerate(new_dataset)},
                                       "item_arr": item_arr,
                                       "varRatios_arr":varRatios_arr})

                self.update_train_index(cur_indices)

            elif dete_method == "STD": # mean-STD
                item_arr = np.array(score_arr)
                mean_STD_arr = [mean_STD(item) for item in item_arr]
                arg = np.argsort(mean_STD_arr)[-acquire_document_num:]  # mean-STD 最大的几个样本的id
                cur_indices = set()
                for i in arg:
                    cur_indices.add(new_datapoints[i])
                self.update_train_index(cur_indices)

            else:
                assert False #"Not Programmed"

        print('dete time consuming： %d seconds:' % (time.time() - tm))

    def get_dete_with_feature(self, dataset, model_path, acquire_document_num,
               model_name='', dete_method = "VRS", subMethod="feature" , returned=False, thisround=-1,):
        tm = time.time()

        print('dete : preparing feature data', end='')
        sample_feature = self.getSimilarityMatrix(dataset, model_path, model_name, feature_only=True) #原始数据的特征
        sample_feature_generated,sample_score_generated = self.getSimilarityMatrix(
            self.generated_train_data, model_path, model_name, feature_only=True, with_predict=True)

        # 对每个样本和它的生成数据之间的feature距离 dis = 1-cos(ori,gene) 样本间越近越大
        # cos_similarity 会得到一个2*2的对称矩阵，返回距离，我们只需要反对角线上的值
        generated_feature_cos_distance = [
            1 - sum(
                [cosine_similarity( [sample_feature[i], sample_feature_generated[i*self.generated_per_sample+j]] )[0][1]
                 for j in range(self.generated_per_sample)]
            ) / self.generated_per_sample for i in range(len(sample_feature))
        ]

        model = torch.load(model_path)
        model.train(False)  # 保持 dropout 关闭
        # data without id
        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]
        # id that not in train_index
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]
        print('dete : preparing batch data', end='')
        data_batches = create_batches(new_dataset, batch_size=self.batch_size, order='no')
        score_arr = []  # 获取模型对未标记样本的输出

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

        if dete_method == "PDVRS":
            item_arr = np.array(score_arr)
            item_arr_generated = np.array(sample_score_generated)
            item_arr_generated = [ np.average(item_arr_generated[oneid*3:oneid*3+3],axis=0) for oneid in new_datapoints ]
            assert len(item_arr)==len(item_arr_generated)
            pd_arr = [allSubMethod.getDeteMethodByName("pd")(x,y) for x,y in zip(item_arr,item_arr_generated)]

            varRatios_arr = list()
            for overAllGroundTruth in item_arr:
                positive_num = np.sum(overAllGroundTruth > 0.5)
                if positive_num == 0:
                    positive_num = 1
                elif positive_num == overAllGroundTruth.size:
                    positive_num = overAllGroundTruth.size - 1
                sorted_overAllGroundTruth = sorted(overAllGroundTruth)
                positive_item = sorted_overAllGroundTruth[-positive_num:]
                varRatios_arr.append(1 - np.mean(positive_item))

            # PDVRS
            # new_score_arr = [(pd_arr[i]*0.5 + varRatios_arr[i])  for i in range(len(varRatios_arr))]
            # PDVRS2
            # new_score_arr = [(pd_arr[i] * 0.25 + varRatios_arr[i]) for i in range(len(varRatios_arr))]
            # PDVRS3
            new_score_arr = [(pd_arr[i] * 0.5 + (varRatios_arr[i] if varRatios_arr[i]<0.5 else 1+varRatios_arr[i])   )
                                for i in range(len(varRatios_arr))]

        if dete_method == "VRS":  # var Ratios
            item_arr = np.array(score_arr)
            varRatios_arr = list()
            for overAllGroundTruth in item_arr:
                positive_num = np.sum(overAllGroundTruth > 0.5)
                if positive_num == 0:
                    positive_num = 1
                elif positive_num == overAllGroundTruth.size:
                    positive_num = overAllGroundTruth.size - 1
                sorted_overAllGroundTruth = sorted(overAllGroundTruth)
                positive_item = sorted_overAllGroundTruth[-positive_num:]
                varRatios_arr.append(1 - np.mean(positive_item))

            # vrs
            new_score_arr = varRatios_arr
            # fvrs
            # new_score_arr = [ (generated_feature_cos_distance[new_datapoints[i]])*varRatios_arr[i]
            #                     for i in range(len(varRatios_arr)) ]
            # fvrs2
            # new_score_arr = [0.1*(generated_feature_cos_distance[new_datapoints[i]]) + varRatios_arr[i]
            #                  for i in range(len(varRatios_arr))]
            # fvrs3
            # new_score_arr = [(generated_feature_cos_distance[new_datapoints[i]]) + varRatios_arr[i]
            #                  for i in range(len(varRatios_arr))]
            # fvrs4
            # new_score_arr = [(generated_feature_cos_distance[new_datapoints[i]]) * 10 + varRatios_arr[i]
            #                  for i in range(len(varRatios_arr))]
            # fvrs5
            # new_score_arr = [(generated_feature_cos_distance[new_datapoints[i]]) * 0.4 + varRatios_arr[i]
            #                  for i in range(len(varRatios_arr))]

        arg = np.argsort(new_score_arr)[-acquire_document_num:]  # ratios 最大的几个样本的id
        cur_indices = set()
        for i in arg:
            cur_indices.add(new_datapoints[i])

        self.savedData.append({"added_index": cur_indices,
                               "index2id": {_index: p[3] for _index, p in enumerate(new_dataset)},
                               "item_arr": item_arr,
                               "varRatios_arr": varRatios_arr,
                               "new_score_arr":new_score_arr,
                               "generated_feature_cos_distance":generated_feature_cos_distance,
                               "sample_score_generated":sample_score_generated})
        self.update_train_index(cur_indices)

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

    def getSimilarityMatrix(self, dataset, model_path='', model_name='', batch_size=800,
                            feature_only=False, with_predict=False):
        '''
        :param feature_only: 表示返回特征还是相似度矩阵
        '''

        model = torch.load(model_path)
        model.train(False)

        # 对剩余样本池创建batch
        data_batches = create_batches(dataset, batch_size=batch_size, order='no')

        temp_feature = []
        temp_score = []
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
                # 2020 09 02 又改回来啦
                output,output_score = model.features(X,with_forward=with_predict)
                output_score = torch.sigmoid(output_score).data.cpu().numpy().tolist() if with_predict else list()
                # output = model.features_with_pred(X)
            temp_feature.extend(output.data.cpu().numpy().tolist())
            _ = temp_score.extend(output_score) if with_predict else list()

        features = np.stack(temp_feature, axis=0)
        scores = np.stack(temp_score, axis=0) if with_predict else list()

        if feature_only:
            if with_predict:
                return features,scores
            else:
                return features

        similarity = cosine_similarity(features) + 1
        return similarity

    def getSimilarityMatrixNTimes(self, dataset, model_path='', model_name='', batch_size=800,
                                  feature_only=False, with_predict=False, nsamp = 100):
        '''
        using for no-dete method: 获取模型对样本的predict和feature
        :param feature_only: 表示返回特征还是相似度矩阵
        '''

        model = torch.load(model_path)
        model.train(True)

        # 对剩余样本池创建batch
        data_batches = create_batches(dataset, batch_size=batch_size, order='no')

        temp_feature_arr = []
        temp_score_arr = []
        for iter_batch,data in enumerate(data_batches):
            print("getting Similarity {}/{}".format(iter_batch,len(data_batches)),end="\r")
            batch_data_numpy  = data['data_numpy']

            X = batch_data_numpy[0]
            Y = batch_data_numpy[1]

            if self.usecuda:
                X = Variable(torch.from_numpy(X).long()).cuda(self.cuda_device)
            else:
                X = Variable(torch.from_numpy(X).long())

            output_arr = []
            output_score_arr = []
            for itr in range(nsamp):

                if model_name == 'BiLSTM':
                    # output = model(words_q, words_a, wordslen_q, wordslen_a)
                    pass
                elif model_name == 'CNN':
                    output, output_score = model.features(X, with_forward=with_predict)

                    output = output.data.cpu().numpy() # feature
                    output_score = torch.sigmoid(output_score).data.cpu().numpy() if with_predict else list() # score


                output_arr.append(output)
                output_score_arr.append(output_score)


            # reshape the two arr into size: btach_size * nsample * nlabel
            # output_arr = [[ output_arr[n][m] for n in range(nsamp) ] for m in range(len(Y)) ]
            # output_score_arr = [[ output_score_arr[n][m] for n in range(nsamp) ] for m in range(len(Y)) ]
            output_arr = np.stack(output_arr, axis=1)
            output_score_arr = np.stack(output_score_arr, axis=1) if with_predict else list()

            # save outputs for each batch
            temp_feature_arr.append(output_arr)
            temp_score_arr.append(output_score_arr)

        features = np.vstack(temp_feature_arr)
        scores = np.vstack(temp_score_arr) if with_predict else list()

        if feature_only:
            if with_predict:
                return features,scores
            else:
                return features

        # similarity = cosine_similarity(features) + 1
        # return similarity


    def obtain_data(self, data, model_path=None, model_name=None, acquire_num=2,
                    method='random', sub_method='', round = 0):

        print("sampling method：" + sub_method,"; use generated data:",self.using_generated_data)
        self.round = round

        # 待完善此处的逻辑
        # self.deal_generated_train_index(method="rdr",para=0.5)

        if model_path == "":
            print("First round of sampling")
            self.get_random(data, acquire_num)
        else:
            if method == 'random':
                self.get_random(data, acquire_num)
            elif method == 'dete':
                if sub_method=="VRS_feature":
                    dete_method,sub_method = sub_method.split("_")
                    self.get_dete_with_feature(data,model_path,acquire_num,model_name,dete_method=dete_method,thisround=round)
                elif sub_method=="PDVRS":
                    self.get_dete_with_feature(data, model_path, acquire_num, model_name, dete_method=sub_method,thisround=round)
                else:
                    self.get_dete(data,model_path,acquire_num,model_name,dete_method=sub_method,thisround=round)
            elif method == 'no-dete': # Bayesian neural network based method
                if sub_method[0] == 'G':
                    self.get_GRKL(data, model_path, acquire_num, rklNo=sub_method[1:], model_name=model_name, thisround=round)
                elif sub_method == 'DAL':
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
                elif sub_method == "VRS":
                    self.get_RKL(data, model_path, acquire_num, rklNo='vrs', model_name=model_name, thisround=round)
                elif sub_method == "VRL":
                    self.get_RKL(data, model_path, acquire_num, rklNo='vrl', model_name=model_name, thisround=round)
                elif sub_method == "LIW":
                    self.get_RKL(data, model_path, acquire_num, rklNo='liw', model_name=model_name, thisround=round, rklMod=0)
                elif sub_method == "LIW1":
                    self.get_RKL(data, model_path, acquire_num, rklNo='liw', model_name=model_name, thisround=round, rklMod=1)
                elif sub_method == "LIW2":
                    self.get_RKL(data, model_path, acquire_num, rklNo='liw', model_name=model_name, thisround=round, rklMod=2)
                elif sub_method == "LIW3":
                    self.get_RKL(data, model_path, acquire_num, rklNo='liw', model_name=model_name, thisround=round, rklMod=3)
                elif sub_method == "VL":
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round, rklMod=(0.0,0.5,0.5))
                elif sub_method == "MVL":
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round)
                elif sub_method == "MVL1":
                    delt = 0.02*round
                    delt = 0.3 if delt > 0.3 else delt
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round, rklMod=(0.4+2*delt,0.3-delt,0.3-delt))
                elif sub_method == "MVL2":
                    delt = 0.02*round
                    delt = 0.3 if delt > 0.3 else delt
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round, rklMod=(0.4,0.3+delt,0.3-delt))
                elif sub_method == "MVL3":
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round, rklMod=(0.4,0, 0.3,0, 0.3,0 ))
                elif sub_method == "MVL4":
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round, rklMod=(0.4, 0.3, 0, 0.3, 0))
                elif sub_method == "MVL5":
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round, rklMod=(0.4,0, 0.3,0, 0,0 ))
                elif sub_method == "MVL6":
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round, rklMod=(0.4,0, 0.3,0.3, 0,0 ))
                elif sub_method == "MVL7":
                    self.get_RKL(data, model_path, acquire_num, rklNo='mvl', model_name=model_name, thisround=round, rklMod=(0.4,0.4, 0.3,0.3, 0.3,0))
                elif sub_method == "FVA":
                    self.get_RKL(data, model_path, acquire_num, rklNo='fva', model_name=model_name, thisround=round )
                elif sub_method == "FVA1":
                    self.get_RKL(data, model_path, acquire_num, rklNo='fva', model_name=model_name, thisround=round, rklMod=1)
                elif sub_method == "FVA2":
                    self.get_RKL(data, model_path, acquire_num, rklNo='fva', model_name=model_name, thisround=round, rklMod=2)





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

                elif sub_method == "RKL8.0":
                    # 233
                    self.get_RKL(data, model_path, acquire_num, rklNo=8, rklMod=0, model_name=model_name, thisround=round)
                elif sub_method == "RKL8.1":
                    # 233
                    self.get_RKL(data, model_path, acquire_num, rklNo=8, rklMod=1, model_name=model_name, thisround=round)
                elif sub_method == "RKL8.2":
                    # 233
                    self.get_RKL(data, model_path, acquire_num, rklNo=8, rklMod=2, model_name=model_name, thisround=round)
                elif sub_method == "RKL8.3":
                    # 233
                    self.get_RKL(data, model_path, acquire_num, rklNo=8, rklMod=3, model_name=model_name, thisround=round)
                elif sub_method == "RKL8.4":
                    # 233
                    self.get_RKL(data, model_path, acquire_num, rklNo=8, rklMod=4, model_name=model_name, thisround=round)
                elif sub_method == "RKL8.5":
                    # 233
                    self.get_RKL(data, model_path, acquire_num, rklNo=8, rklMod=5, model_name=model_name, thisround=round)
                elif sub_method == "RKL8.6":
                    # 233
                    self.get_RKL(data, model_path, acquire_num, rklNo=8, rklMod=6, model_name=model_name, thisround=round)

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
                # elif sub_method == "dsm3RKL4":
                #     _, unlabeled_index = self.get_RKL(data, model_path, acquire_num*max(1.0,(12-1.5*round)), model_name=model_name, thisround=round, returned=True)
                #     self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path, model_name=model_name)
                # elif sub_method == "dsm4RKL4":
                #     _, unlabeled_index = self.get_RKL(data, model_path, acquire_num*max(1.0,(3-0.2*round)), model_name=model_name, thisround=round, returned=True)
                #     self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path, model_name=model_name)
                # elif sub_method == "dsm5RKL4":
                #     temp = [2,2,2,2,2,2,2,2,2,2, 1.5,1.5,1.5,1.5,1.5, 1,1,1,1,1,1,1,1,1,1 ]
                #     _, unlabeled_index = self.get_RKL(data, model_path, acquire_num*temp[round], model_name=model_name, thisround=round, returned=True)
                #     self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path, model_name=model_name)
                # elif sub_method == "dsm6RKL4":
                #     temp = [2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1]
                #     _, unlabeled_index = self.get_RKL(data, model_path, acquire_num * temp[round],
                #                                       model_name=model_name, thisround=round, returned=True)
                #     self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                #                         model_name=model_name)
                # elif sub_method == "dsm7RKL4":
                #     temp = [2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4,  2, 2, 2, 2, 1.5, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1]
                #     _, unlabeled_index = self.get_RKL(data, model_path, acquire_num * temp[round],
                #                                       model_name=model_name, thisround=round, returned=True)
                #     self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                #                         model_name=model_name)
                # elif sub_method == "dsm8RKL4":
                #     temp = [2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4,4,4,4 , 4,4,4,4, 4 ]
                #     _, unlabeled_index = self.get_RKL(data, model_path, acquire_num * temp[round],
                #                                       model_name=model_name, thisround=round, returned=True)
                #     self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                #                         model_name=model_name)
                # elif sub_method == "dsm9RKL4":
                #     _, unlabeled_index = self.get_RKL(data, model_path, acquire_num * max((0.25*round),1),
                #                                       model_name=model_name, thisround=round, returned=True)
                #     self.get_submodular(data, unlabeled_index, acquire_num, model_path=model_path,
                #                         model_name=model_name)

                elif sub_method == "dsm1FEL":

                    _, unlabeled_index = self.get_FEL(data, model_path, acquire_num * 2, model_name=model_name,
                                                      thisround=round, returned=True)

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
                elif sub_method == "FERKfpL":
                    self.get_FEL_RKL_ETL(data, model_path,acquire_num,model_name=model_name,combine_method="FERKfpL")
                    #
                elif sub_method == "FELplusRKL":
                    # # # 普通
                    self.get_FELplusRKL(data, model_path, acquire_num, model_name=model_name, thisround=round)
                elif sub_method == "FEVRL":
                    self.get_FEL_RKL_ETL(data, model_path,acquire_num,model_name=model_name,combine_method="FEVRL")
                    pass
                elif sub_method == "FEVRS":
                    self.get_FEL_RKL_ETL(data, model_path,acquire_num,model_name=model_name,combine_method="FEVRS")
                    pass
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


