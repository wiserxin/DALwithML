# !/bin/bash

rm res.zip
zip res.zip ./result/*.txt
python ezEmail.py --From wiserxin_kindle@163.com --To flowing_clouds_xy@163.com --pw 111111abc --file_path=res.zip --header=visionData --body=qaq

# 检查是否成功发送
if [ $?==0 ];then
    rm ./result/*.txt
else
    echo "发送失败,仍需测试"
fi

