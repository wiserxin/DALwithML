#!/bin/bash

rm result.txt
git pull
git checkout master
CUDA_VISIBLE_DEVICES=0 python main.py --learning_rate=4e-3
