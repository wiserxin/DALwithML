#!/bin/sh

git pull
CUDA_VISIBLE_DEVICES=0 python main.py
