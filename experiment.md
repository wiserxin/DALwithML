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


## TA实验
使用textAttack包生成数据进行数据增强/数据扩增的主动学习方向探索
实验数据在 [mlabs]TA+stack 中存储与展示

### 每个样本使用几个扩增数据的探索
TA1，TA2，TA3使用1、2、3个进行测试，
综合来看TA2效果较好。
发现总数据量少的时候，使用的扩增样本越多效果越好(3)，
而数据量多时，使用几个扩增样本并没有本质的区别。
此处与EDA论文里的结论相符  
EDA- Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks.pdf

###  数据扩增时的参数问题
对比样本内替换率为10%30%50%的三组实验，可以发现替换率为10%的效果最好

### 一些复杂策略的探索

#### rdr
random drop out , random dropout selected self.generated_train_index in rate para
随每个round，随即删除一定比例的生成数据。
使用速率0.5实验了一下，效果不好，可以看到vrs-rdr0.5和vrs基本不相上下。
猜测是在主动学习的前期生成的数据更重要一些，能够更好的提升模型的性能。

#### esd
easy slow down  
`self.generated_used_per_sample = max(0,self.generated_per_sample-self.round//4)`  
在前期使用所有的生成数据（3个），每过4个轮次，减少1个  
前期略有优势(3个)的使用在前期效果不错。
但是实质上与TA2相比并无本质区别，不过可以作为节约算力的方法引入。

#### fvrs
vrs with feature distance   
- fvrs/fvrs1  
    cos_distance * vrs 
- fvrs2   
    0.1 * cos_distance + vrs
- fvrs3  
    cos_distance + vrs
- fvrs4   
    10 * cos_distance + vrs

