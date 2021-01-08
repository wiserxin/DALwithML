#!/bin/bash

echo "Running for single labels datasets now ..."
rm result.txt
git pull
git checkout single_label_experiments
git pull
CUDA_VISIBLE_DEVICES=0 python main.py --learning_rate=4e-3
