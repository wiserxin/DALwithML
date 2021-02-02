# 实验记录

## VRS实验
最简单的方法，即使用 mean(1-pos) 来选取主动学习样本。

在multilabel classification实验中效果貌似也是最好的？


## MVL 相关实验

### 方法简介 
MVL使用的是基于 均值（Mean）方差（Var）的损失（Loss）计算。  
对不同的值进行加权求和
- 1 均值 w_m
    - (mean(pos)-mean(neg))**2
- 2 方差  
    - 组间方差 w_m1  
        var1(pos)+var1(neg)  
        var1: 认为是多次试验的随机误差，即dropout引起的离差平方和  
        实验中体现为模型的鲁棒性？  
    - 组内方差 w_m2
        var2(pos)+var2(neg)  
        var2: 认为是描述因素各个水平效应的影响，实验中体现为  
        pos label 间的差异程度、neg label 间的差异程度  
        
### 实验与分析

#### MVL
使用默认权重(0.4,0.3,0.3)  
效果较优，前2次采样接近VRS,之后稳定落后VRS方法越0.01直至结束。  
MVL前期较优，ferkl后期较优，

#### VL
使用权重(0,0.5,0.5)
效果较差，与rkl方法效果近似。
VL仅考察方差，rkl仅考察均值。

#### MVL1
使用动态权重(0.4+2*delt,0.3-delt,0.3-delt)  
探索 均值与方差在时间上的重要性变化
```python
    delt = 0.02*round
    delt = 0.3 if delt > 0.3 else delt
    weight = (0.4+2*delt,0.3-delt,0.3-delt)
```
    

#### MVL2
使用动态权重(0.4,0.3+delt,0.3-delt)  
探索 分解的方差在时间上的重要性变化
```python
    delt = 0.02*round
    delt = 0.3 if delt > 0.3 else delt
    weight = (0.4,0.3+delt,0.3-delt)
```

#### MVL3 
使用权重(0.4,0, 0.3,0, 0.3,0 )  
仅考察pos的信息,观察效果  
(1-pos_m) + pos_s1 + pos_s2  
与MVL4相比去掉了neg_mean的考察  

#### MVL4 
使用权重(0.4, 0.3,0, 0.3,0 )  
与MVL相比去掉了neg_s1,neg_s2的考察

#### MVL5 
使用权重(0.4,0, 0.3,0, 0,0 )  
仅考察pos的信息,观察效果  
(1-pos_m) + pos_s1  
与MVL3相比去掉了pos_s2的考察  

#### MVL6 
使用权重(0.4,0, 0.3,0.3, 0,0 )    
(1-pos_m) + pos_s1 + neg_s1  
与MVL5相比增加了neg_s1的考察  
此时更关注该样本在dropout下的扰动情况

#### MVL7
(0.4,0.4, 0.3,0.3, 0.3,0)
(1-pos_m) + neg_m + pos_s1 + neg_s1 + pos_s2  
仅不考察neg_s2,即neg label不同造成的扰动