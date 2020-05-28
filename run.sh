#!/bin/sh

rm output.txt
git pull
CUDA_VISIBLE_DEVICES=1 python main.py
