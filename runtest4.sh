#!/bin/sh

git pull
git checkout test_branch_for-selected_data
CUDA_VISIBLE_DEVICES=1 python test4.py --sampling_batch_size=2048
