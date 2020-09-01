#!/bin/bash

# change learning rate for xml cnn
rm result.txt
git pull
git checkout test_new_eval_el
git pull
CUDA_VISIBLE_DEVICES=0 python main.py --learning_rate=1e-3
