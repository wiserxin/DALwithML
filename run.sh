#!/bin/sh

rm result.txt
git pull
CUDA_VISIBLE_DEVICES=1 python main.py --learning_rate=0.0005
