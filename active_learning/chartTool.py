# -*- coding: UTF-8 -*-
import json
import urllib3, urllib


global group_name
group_name = "[5.25-?]CNN+PFS-18000+MRR+5000+150"

'''
No-Deterministic
Deterministic
Random
'''
global sample_method
sample_method = "Random+seed-0"

global max_performance
max_performance = "0"

global bar_legend
bar_legend = 0


# Update line chart
def updateLineChart(mrr, sm, gp_name = group_name, max = max_performance):

    http = urllib3.PoolManager()

    global sample_method
    global max_performance

    if max == 0:
        max = max_performance

    info = {}
    info['active_combine'] = gp_name
    info['sample_method'] = sm
    info['max_performance'] = max
    info['bar_legend'] = bar_legend
    info["mrr_data"] = mrr
    info["type"] = 1

    refs = urllib.parse.urlencode(info)

    urls = '10.2.26.117/updateChart/?' + refs
    http.request('GET', urls)

# Update histogram
def updateBarChart(data):

    http = urllib3.PoolManager()

    global active_combine
    global sample_method

    info = {}
    info['active_combine'] = active_combine
    info['sample_method'] = sample_method
    info['max_performance'] = max_performance
    info['bar_legend'] = bar_legend
    info["data"] = data
    info["type"] = 2

    refs = urllib.parse.urlencode(info)

    urls = '10.2.26.117/updateChart/?' + refs
    http.request('GET', urls)


import os
def walkFile(path):
    txt = []
    for root, dirs, files in os.walk(path):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            temp = os.path.abspath(os.path.join(root, f))
            if temp[-4:] == '.txt':
                txt.append(temp)
    #                 print(temp)

    #         遍历所有的文件夹
    #         for d in dirs:
    #             print(os.path.join(root, d))
    return txt


class VisionResult():
    def __init__(self, filepath):
        fileContent = []
        with open(filepath, 'r') as f:
            fileContent = f.readlines()

        self.group_name, self.sample_method, self.round = fileContent[0][:-1].split('\t')
        self.round = int(self.round)

        self.performance = []
        count = 0
        while (count < self.round):
            try:
                self.performance.append(float(fileContent[count + 1].split('\t')[1]))
                count += 1
            except:
                break
        print("{}:{} - Round{}:{}".format(self.group_name, self.sample_method, len(self.performance), self.round))

    def send(self,max_performance=0.90):
        for i in self.performance:
            updateLineChart(str(i), self.sample_method, gp_name=self.group_name, max=max_performance)


if __name__ == '__main__':
    for txt in walkFile(r'../result/'):
        perf = VisionResult(txt)
        perf.send(max_performance=0.90)
