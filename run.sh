#!/bin/sh

rm output.txt
git pull
CUDA_VISIBLE_DEVICES=1 python main.py --learning_rate=0.0001
CUDA_VISIBLE_DEVICES=1 python main.py --learning_rate=0.0005
CUDA_VISIBLE_DEVICES=1 python main.py --learning_rate=0.001
