#!/bin/bash

rm result.txt
git pull
git checkout test_new_eval_el
CUDA_VISIBLE_DEVICES=0 python main.py --learning_rate=4e-3
