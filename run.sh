#!/bin/bash

rm result.txt
git pull
git checkout master
CUDA_VISIBLE_DEVICES=0 python main.py --learning_rate=4e-3
rm res.zip
zip res.zip ./result/*.txt
python ezEmail.py --From wiserxin_kindle@163.com --To flowing_clouds_xy@163.com --pw 111111abc --file_path=res.zip --header=visionData --body=qaq
