# coding=utf-8
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
    parser.add_argument('--num_epochs', type=int, default=40, help='training epoch')
    parser.add_argument('--use_pretrained_word_embedding', type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=800, help='')
    parser.add_argument('--sampling_batch_size', type=int, default=800, help='')
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

        {
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]KIM+DAL+1e4trn",
        #     "max_performance": 0.90,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "random",
        #     "sub_acquire_method": "",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 20,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 400,
        #     "warm_start_random_seed": 0,
        #     "sample_method": "Random+400*20-800b-0",
        # },{
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]KIM+DAL+1e4trn",
        #     "max_performance": 0.90,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "random",
        #     "sub_acquire_method": "",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 20,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 400,
        #     "warm_start_random_seed": 16,
        #     "sample_method": "Random+400*20-800b-16",
        # },{
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]KIM+DAL+1e4trn",
        #     "max_performance": 0.90,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "random",
        #     "sub_acquire_method": "",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 20,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 400,
        #     "warm_start_random_seed": 32,
        #     "sample_method": "Random+400*20-800b-32",
        # },{
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]KIM+DAL+1e4trn",
        #     "max_performance": 0.90,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "random",
        #     "sub_acquire_method": "",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 20,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 400,
        #     "warm_start_random_seed": 64,
        #     "sample_method": "Random+400*20-800b-64",
        # },
        #
        # {
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]KIM+DAL+1e4trn",
        #     "max_performance": 0.90,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "no-dete",
        #     "sub_acquire_method": "DAL",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 20,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 400,
        #     "warm_start_random_seed": 0,
        #     "sample_method": "No-Deterministic+DAL-400*20-800b+0",
        # },{
        #     "model_name": "CNN",
        #     "group_name": "[mlabs]KIM+DAL+1e4trn",
        #     "max_performance": 0.90,
        #     "data_path": "../../datasets/rcv2/",
        #     "acquire_method": "no-dete",
        #     "sub_acquire_method": "DAL",
        #     "unsupervised_method": 'submodular',
        #     "submodular_k": 2,
        #     "num_acquisitions_round": 20,
        #     "init_question_num": 400,
        #     "acquire_question_num_per_round": 400,
        #     "warm_start_random_seed": 16,
        #     "sample_method": "No-Deterministic+DAL-400*20-800b+16",
        # },{
            "model_name": "CNN",
            "group_name": "[mlabs]KIM+DAL+1e4trn",
            "max_performance": 0.90,
            "data_path": "../../datasets/rcv2/",
            "acquire_method": "no-dete",
            "sub_acquire_method": "BEL",
            "unsupervised_method": 'submodular',
            "submodular_k": 2,
            "num_acquisitions_round": 20,
            "init_question_num": 400,
            "acquire_question_num_per_round": 400,
            "warm_start_random_seed": 0,
            "sample_method": "No-Deterministic+BEL-400*20-800b+0",
        },{
            "model_name": "CNN",
            "group_name": "[mlabs]KIM+DAL+1e4trn",
            "max_performance": 0.90,
            "data_path": "../../datasets/rcv2/",
            "acquire_method": "no-dete",
            "sub_acquire_method": "BEL",
            "unsupervised_method": 'submodular',
            "submodular_k": 2,
            "num_acquisitions_round": 20,
            "init_question_num": 400,
            "acquire_question_num_per_round": 400,
            "warm_start_random_seed": 64,
            "sample_method": "No-Deterministic+BEL-400*20-800b+64",
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
        # visual_data_path = os.path.join("result", sample_method + ".txt")

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

        train_data  = train_data[:10000]
        val_data    = val_data[:2000]

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

        # with open(visual_data_path, 'a') as f:
        #     print(config["group_name"],sample_method,num_acquisitions_round,sep='\t',file=f)

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

            if "group_name" in config:
                updateLineChart(str(test_performance), sample_method, gp_name=config["group_name"], max=max_performance)
            else:
                updateLineChart(str(test_performance), sample_method, max=max_performance)

            method_result.append(test_performance)

            # if not os.path.exists(visual_data_path): # 被zip.sh删掉了,需要重新创建，并写头信息
            #     with open(visual_data_path, 'a') as f:
            #         print(config["group_name"], sample_method, num_acquisitions_round, sep='\t', file=f)
            # with open(visual_data_path, 'a') as f:
            #     print("acq round {} : \t {}"
            #           .format(i,test_performance),
            #           file=f)

        print("acquire_method: {}，sub_acquire_method: {}, warm_start_random_seed{}"
              .format(acquire_method, sub_acquire_method, warm_start_random_seed))
        print(method_result)
        # with open(visual_data_path,'a') as f:
        #     print("acquire_method: {},sub_acquire_method: {}, warm_start_random_seed{}"
        #           .format(acquire_method, sub_acquire_method, warm_start_random_seed),
        #           file=f )
        #     print(method_result, file=f )
        #     print('', file=f)

        allMethods_results.append(method_result)
        shutil.rmtree(checkpoint_path)
        with open(config["group_name"]+sample_method.split('+')[1].split('-')[0]+"_detail.pkl",'wb') as f:
            pkl.dump(acquisition_function.savedData, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
