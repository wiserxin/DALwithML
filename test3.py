# coding=utf-8

# 小样本验证 有效性
from __future__ import print_function
import numpy as np
import torch

import os
import shutil
import pickle as pkl
import argparse

from neural.util import Trainer, Loader
from neural.models import BiLSTM
from neural.models import CNN

from active_learning.acquisition import Acquisition
from active_learning.chartTool import *

torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')

    parser.add_argument('--answer_count', type=int, default=5, help='the amount of answer for each quesiton')
    parser.add_argument('--num_epochs', type=int, default=20, help='training epoch')
    parser.add_argument('--use_pretrained_word_embedding', type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--sampling_batch_size', type=int, default=60, help='')
    parser.add_argument('--with_sim_feature', type=bool, default=True, help='whether use sim_feature in deep model')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='')
    parser.add_argument('--pretrained_word_embedding', default="../../datasets/rcv2/glove.6B.300d.txt", help='')
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--word_hidden_dim', type=int, default=75, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='')
    parser.add_argument('--target_size', type=int, default=103, help='rcv2:103 ')
    parser.add_argument('--top_k', type=int, default=40, help='rcv2:40 , eurLex:100')
    parser.add_argument('--word_out_channels', type=int, default=200, help='')
    parser.add_argument('--result_path', default="result/rcv2/",help='')
    parser.add_argument('--device', type=int, default=[0], help='')
    parser.add_argument('--cal_Aleatoric_uncertainty', type=bool, default=False, help='')
    parser.add_argument('--sampling_number', type=int, default=1, help='')

    args = parser.parse_args()
    return args


####################################################################################################
#############################              active learning               ###########################
def main(args):

    task_seq = [
        # The config for a task:
        # acquire_method(sub_acquire_method): random(""), no-dete("DAL","BALD"), dete("coreset","entropy",...)
        # "../../datasets/answer_selection/YahooCQA/data/data-FD/"

        # {
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]CNN+DAL+3e4trn",
        #     "max_performance": 0.80,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "random",
        #     "sub_acquire_method": "",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 40,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 200,
        #     "warm_start_random_seed": 0,
        #     "sample_method": "Random+40-0",
        # }, {
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]CNN+DAL+3e4trn",
        #     "max_performance": 0.80,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "random",
        #     "sub_acquire_method": "",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 40,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 200,
        #     "warm_start_random_seed": 16,
        #     "sample_method": "Random+40-16",
        # }, {
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]CNN+DAL+3e4trn",
        #     "max_performance": 0.80,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "random",
        #     "sub_acquire_method": "",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 40,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 200,
        #     "warm_start_random_seed": 32,
        #     "sample_method": "Random+40-32",
        # },{
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]CNN+DAL+3e4trn",
        #     "max_performance": 0.80,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "random",
        #     "sub_acquire_method": "",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 40,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 200,
        #     "warm_start_random_seed": 64,
        #     "sample_method": "Random+40-64",
        # },

        {
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]CNN+DAL+3e4trn",
        #     "max_performance": 0.80,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "no-dete",
        #     "sub_acquire_method": "DAL",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 40,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 200,
        #     "warm_start_random_seed": 0,
        #     "sample_method": "No-Deterministic+DAL40+0",
        # },{
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]CNN+DAL+3e4trn",
        #     "max_performance": 0.80,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "no-dete",
        #     "sub_acquire_method": "DAL",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 40,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 200,
        #     "warm_start_random_seed": 16,
        #     "sample_method": "No-Deterministic+DAL40+16",
        # },{
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]CNN+DAL+3e4trn",
        #     "max_performance": 0.80,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "no-dete",
        #     "sub_acquire_method": "DAL",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 40,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 200,
        #     "warm_start_random_seed": 32,
        #     "sample_method": "No-Deterministic+DAL40+32",
        # },{
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]CNN+DAL+3e4trn",
        #     "max_performance": 0.80,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "no-dete",
        #     "sub_acquire_method": "DAL",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 40,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 200,
        #     "warm_start_random_seed": 64,
        #     "sample_method": "No-Deterministic+DAL40+64",
        # },


            "model_name": "CNN",
            "group_name": "[mlabs]CNN+DAL+3e4trn",
            "max_performance": 0.80,
            "data_path": "../../datasets/rcv2/",
            "acquire_method": "random",
            "sub_acquire_method": "DAL",
            "unsupervised_method": 'submodular',
            "submodular_k": 2,
            "num_acquisitions_round": 10,
            "init_question_num": 20,
            "acquire_question_num_per_round": 20,
            "warm_start_random_seed": 64,
            "sample_method": "No-Deterministic+DAL40+64",
        },
        {  "model_name": "CNN",
            "group_name": "[mlabs]CNN+DAL+3e4trn",
            "max_performance": 0.80,
            "data_path": "../../datasets/rcv2/",
            "acquire_method": "no-dete",
            "sub_acquire_method": "DAL",
            "unsupervised_method": 'submodular',
            "submodular_k": 2,
            "num_acquisitions_round": 10,
            "init_question_num": 20,
            "acquire_question_num_per_round": 20,
            "warm_start_random_seed": 64,
            "sample_method": "No-Deterministic+DAL40+64",
        },

    ]

    allMethods_results = []   #Record the performance results of each method during active learning

    for config in task_seq:

        print("-------------------{}-{}-------------------".format(config["group_name"], config["sample_method"]))

        ####################################### initial setting ###########################################
        data_path = config["data_path"]
        model_name = config["model_name"] if "model_name" in config else 'CNN'
        num_acquisitions_round = config["num_acquisitions_round"]
        acquire_method = config["acquire_method"]
        sub_acquire_method = config["sub_acquire_method"]
        init_question_num = config["init_question_num"] if "init_question_num" in config else 800 # number of initial training samples
        acquire_question_num_per_round = config["acquire_question_num_per_round"] if "acquire_question_num_per_round" in config else 100 #Number of samples collected per round
        warm_start_random_seed = config["warm_start_random_seed"]  # the random seed for selecting the initial training set
        sample_method = config["sample_method"]

        loader = Loader()

        print('model:', model_name)
        print('dataset:', data_path)
        print('acquisition method:', acquire_method, "+", sub_acquire_method)

        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)

        if not os.path.exists(os.path.join(args.result_path, model_name)):
            os.makedirs(os.path.join(args.result_path, model_name))

        if not os.path.exists(os.path.join(args.result_path, model_name, 'active_checkpoint', acquire_method)):
            os.makedirs(os.path.join(args.result_path, model_name, 'active_checkpoint', acquire_method))

        data = loader.load_rcv2(data_path)

        train_data = data['train_points']
        val_data = data['test_points']

        trn_data_index_1 = {0, 7683, 517, 6662, 4620, 1041, 2065, 5655, 6167, 25, 7705, 7199, 4128, 4130, 6181, 39,
                            5676, 6707, 3637, 2616, 2617, 3644, 5180, 64, 4676, 1100, 1616, 4178, 4180, 4692, 6230,
                            6742, 7768, 3675, 1118, 7776, 5221, 6758, 5740, 110, 2158, 3184, 114, 1650, 3187, 6261,
                            8314, 2173, 4734, 2175, 641, 2182, 3718, 8328, 1164, 2700, 142, 143, 7826, 7827, 8348, 4254,
                            7838, 3233, 8356, 1189, 5285, 2728, 3243, 1708, 4269, 4790, 3255, 1208, 5816, 3260, 7882,
                            5836, 1229, 3278, 3789, 5837, 6872, 1753, 732, 1245, 8412, 5855, 2272, 8415, 1764, 2789,
                            3302, 6373, 1768, 3305, 6375, 3820, 7920, 4852, 8437, 6394, 765, 7422, 2304, 6400, 6912,
                            7939, 267, 2318, 3343, 272, 3855, 787, 7957, 791, 6432, 1826, 6946, 7972, 7975, 808, 8490,
                            3371, 1324, 1837, 1326, 1327, 304, 1329, 4917, 1335, 824, 8509, 4416, 5443, 4421, 6982,
                            1867, 5964, 6476, 6992, 3409, 7505, 4956, 864, 3938, 5990, 8039, 2926, 2928, 3953, 2931,
                            1908, 4469, 5494, 4985, 897, 6530, 2952, 8085, 6550, 7577, 1434, 7582, 8096, 3495, 1962,
                            3505, 3512, 1468, 6077, 3006, 6588, 1474, 6595, 3524, 7107, 1478, 3014, 6088, 2506, 464,
                            7633, 3026, 467, 1498, 4059, 4062, 483, 6115, 7656, 2537, 6636, 7148, 495, 3572, 3063, 3065,
                            2554}
        trn_data_index_2 = {3585, 2052, 4614, 4103, 1548, 4108, 1040, 529, 4117, 1046, 24, 2080, 4641, 2084, 1065, 3118,
                            4147, 1588, 2046, 2102, 2615, 569, 4663, 59, 571, 4669, 1090, 3650, 3651, 3144, 3145, 2122,
                            4682, 79, 1618, 1623, 1115, 1627, 4700, 609, 4198, 2151, 2664, 1131, 2667, 3698, 2676, 1653,
                            3193, 1150, 126, 639, 1156, 133, 4606, 3720, 650, 2186, 4236, 4748, 3219, 4246, 1687, 4770,
                            166, 3750, 1195, 4781, 688, 1718, 188, 700, 1726, 1215, 2238, 2748, 4291, 4292, 2246, 201,
                            1738, 3273, 2252, 4310, 3800, 222, 4832, 3811, 228, 2788, 1256, 2281, 4841, 2798, 3316,
                            2297, 2298, 763, 1276, 2811, 3325, 1279, 3326, 4351, 3330, 4354, 2832, 1814, 2838, 2841,
                            1306, 285, 1313, 803, 1827, 3372, 3888, 4400, 3890, 307, 2357, 822, 1846, 3384, 320, 2374,
                            3910, 2376, 3400, 4427, 3407, 2896, 1876, 4439, 3935, 2400, 1377, 4452, 359, 2409, 1387,
                            2414, 2416, 3952, 4467, 374, 887, 3960, 4475, 2943, 896, 1923, 389, 1933, 1935, 3983, 2451,
                            3987, 2453, 406, 918, 1433, 2458, 2971, 925, 414, 3485, 928, 3498, 1453, 431, 432, 3510,
                            2494, 3520, 1986, 3525, 966, 4041, 1483, 2513, 4563, 3028, 4564, 1495, 2013, 481, 482, 3553,
                            2532, 2534, 489, 1002, 1009, 2548, 4085, 502, 4086, 3069, 510}
        val_data_index_1 = {385, 3, 138, 394, 140, 141, 144, 145, 147, 150, 23, 414, 287, 32, 160, 162, 415, 417, 298,
                            171, 427, 431, 304, 307, 309, 56, 312, 441, 318, 194, 326, 72, 203, 335, 230, 103, 359, 234,
                            115, 253}
        val_data_index_2 = {1409, 1417, 138, 1419, 144, 662, 415, 32, 677, 1445, 1446, 431, 304, 307, 312, 441, 824,
                            830, 1219, 1096, 1353, 1612, 720, 464, 849, 1361, 471, 1240, 857, 477, 1122, 868, 741, 998,
                            359, 878, 625, 498, 1014, 1278}

        def make_dataset_from_index(all_dataset, indexs):
            r = []
            for i in all_dataset:
                if i[3] in indexs:
                    r.append(i)
            return r

        train_data_1 = make_dataset_from_index(train_data, trn_data_index_1)
        train_data_2 = make_dataset_from_index(train_data, trn_data_index_2)
        train_data   = list()
        train_data.extend(train_data_1)
        train_data.extend(train_data_2)
        val_data_1 = make_dataset_from_index(val_data, val_data_index_1)
        val_data_2 = make_dataset_from_index(val_data, val_data_index_2)
        val_data   = list()
        val_data.extend(val_data_1)
        val_data.extend(val_data_2)



        #word embedding
        word_embeds = data['embed'] if args.use_pretrained_word_embedding else None

        word_vocab_size = len(data['vocab'][1])

        print(' The total amount of training data：%d\n' %len(train_data),   # Total number of training samples (number of question answer pair)
              'The total amount of val data：%d\n' %len(val_data),
              'The total amount of test data：%d' %len(val_data))

        acquisition_function = Acquisition(train_data,
                                            seed=warm_start_random_seed,
                                            cuda_device=args.device[0],
                                            batch_size=args.sampling_batch_size,
                                            submodular_k=config["submodular_k"])

        checkpoint_path = os.path.join(args.result_path, 'active_checkpoint', config["group_name"], sample_method)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        method_result = []  # Record the performance results of each method during active learning
        ####################################### acquire data and retrain ###########################################
        for i in range(num_acquisitions_round):

            print("current round：{}".format(i))

            #-------------------acquisition---------------------
            if i == 0:#first round
                acq = init_question_num
                a_m = "random"
                m_p = ""
            else:
                acq = acquire_question_num_per_round
                a_m = acquire_method
                m_p = os.path.join(checkpoint_path, 'modelweights')

            acquisition_function.obtain_data(train_data,
                                             model_path=m_p,
                                             model_name=model_name,
                                             acquire_num=acq,
                                             method=a_m,
                                             sub_method=sub_acquire_method,
                                             unsupervised_method=config["unsupervised_method"],
                                             round = i)

            # -------------------prepare training data---------------------
            '''
            train_data format：
            {
                'str_words_q': str_words_q,  # question word segmentation
                'str_words_a': str_words_a,  # answer word segmentation
                'words_q': words_q,  # question word id
                'words_a': words_a,  # answer word id
                'tag': tag,  # sample tag id
            }
            '''

            new_train_index = (acquisition_function.train_index).copy()
            sorted_train_index = list(new_train_index)
            sorted_train_index.sort()
            labeled_train_data = [train_data[i] for i in sorted_train_index]

            print("Labeled training samples: {}".format(len(acquisition_function.train_index)))

            # -------------------------------------train--------------------------------------


            print('.............Recreate the model...................')
            if model_name == 'BiLSTM':
                    model = BiLSTM(word_vocab_size,
                                   args.word_embedding_dim,
                                   args.word_hidden_dim,
                                   args.target_size,
                                   pretrained=word_embeds,
                                   with_sim_features=args.with_sim_feature,
                                   cuda_device=args.device[0],
                                   )
            if model_name == 'CNN':
                    model = CNN(word_vocab_size,
                                args.word_embedding_dim,
                                args.word_out_channels,
                                args.target_size,
                                pretrained=word_embeds,
                                cuda_device=args.device[0],)

            model.cuda(args.device[0])

            trainer = Trainer(model,
                              args.result_path,
                              model_name,
                              eval_begin=1,
                              cuda_device=args.device[0],
                              top_k= args.top_k
                              )

            test_performance = trainer.train_supervisedLearning(args.num_epochs,
                                                                labeled_train_data,
                                                                val_data,
                                                                args.learning_rate,
                                                                checkpoint_path=checkpoint_path,
                                                                batch_size=args.batch_size
                                                                )

            print('.' * 50)
            print("Test performance: {}".format(test_performance))
            print('*' * 50)

            #--------------------------Send data for a visual web page------------------------------
            max_performance = config["max_performance"] if "max_performance" in config else 0

            # if "group_name" in config:
            #     updateLineChart(str(test_performance), sample_method, gp_name=config["group_name"], max=max_performance)
            # else:
            #     updateLineChart(str(test_performance), sample_method, max=max_performance)

            method_result.append(test_performance)
            # with open('result.txt', 'a') as f:
            #     print("acq round {} : \t {}"
            #           .format(i,test_performance),
            #           file=f)

        print("acquire_method: {}，sub_acquire_method: {}, warm_start_random_seed{}"
              .format(acquire_method, sub_acquire_method, warm_start_random_seed))
        print(method_result)
        # with open('result.txt','a') as f:
        #     print("acquire_method: {}，sub_acquire_method: {}, warm_start_random_seed{}"
        #           .format(acquire_method, sub_acquire_method, warm_start_random_seed),
        #           file=f )
        #     print(method_result, file=f )
        #     print('', file=f)

        allMethods_results.append(method_result)
        shutil.rmtree(checkpoint_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
