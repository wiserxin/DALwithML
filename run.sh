#!/bin/sh

rm result.txt
git pull
git checkout master
CUDA_VISIBLE_DEVICES=1 python main.py --learning_rate=4e-3
