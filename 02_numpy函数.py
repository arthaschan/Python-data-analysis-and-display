# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     numpy的函数
   Author :       Zeke
   date：          2018/5/26
   Description : numpy的函数简介
-------------------------------------------------
"""
import numpy as np

# 1/ 随机数函数
# (1)
# rand(d0,d1,...,dn)  # 根据d0,d1,...,dn创建随机数数组，浮点数，[0,1),均匀分布
# randn(d0,d1,...,dn) # 根据d0,d1,...,dn创建随机数数组,标准正态分布
# randint(low,high,size]) # 根据shape创建随机整数或整数数组，范围是[low,high]
# seed(s) #随机数种子，s是给定的种子值
np.random.seed(10)
print(np.random.rand(3,4,5))
print(np.random.randn(3,4,5))
print(np.random.randint(100,200,(3,4)))

# (2)
# shuffle(a) #根据数组a的第一轴（最外围的轴）进行随机排列，改变数组a
# permutation(a) # 根据数组a的第1轴产生一个新的乱序数组，不改变数组a
# choice(a[,size,replace,p]) # 从1维数组a中以概率p抽取元素，形成size形状新数组，replace表示是否可以重用元素，默认为False
a = np.random.randint(100,200,(3,4))
b = np.random.shuffle(a)
c = np.random.permutation(a)

d = np.random.choice(np.random.randint(1,100,(20, )),
                     (3,2),
                     replace=True)
print(d)


# (3)
# uniform(low,high,size) # 产生均匀分布的数组
# normal(loc, scale, size) # 产生正态分布的数组,loc为均值，scale为标准差，size为形状
# poission(lam,size) #产生具有poission分布的数组，lam为随机事件发生率，size形状
print(np.random.uniform(0, 10, (3, 4)))
print(np.random.normal(10, 5, (3, 4)))



# 2/ 统计函数
# (1)
# sum(a, axis=None)
# mean(a, axis=None)
# average(a, axis=None, weight=None)
# std(a, axis=None)
# std(a, axis=None) # 给定轴计算方差

print(u"统计函数")
A = np.arange(15).reshape(3,5)  #np.arange(15) 得到的是 0 1  ** 14的一维 数组，15个元素，经过reshape后，就调整为 3*5的矩阵
print(A)
print(np.sum(A))
print(np.mean(A, axis=1)) # 最外层维度为3代表axis=0，第二层维度为5代表axis=1，表示将第二层5个元素求平均值
print(np.mean(A, axis=0)) #在第1维度做运算
print(np.average(A, axis=0, weights=[10, 5, 1]))

# (2)
# min(a) max(a) median(a)
# argmin(a) argmax(a) #计算数组a中元素最小值/最大值的降一维后下标
# unravel_index(index, shape) # 根据shape将一维下标index转换成多维下标
# ptp(a) # 计算数组中最大值与最小值的差
B = np.arange(15, 0, -1).reshape(3, 5)
print(np.max(B))
###一般结合使用
print(np.argmax(B))  # 得到的是扁平化后的下标
print(np.unravel_index(np.argmax(B), B.shape)) #根据B的形状重塑成多维下标
####
print(np.ptp(B))
print(np.median(B))


# 3/ 梯度函数
# np.gradient(f)  #计算数组f中元素的梯度，当f为多维度时，返回每个维度的梯度
# 例如XY坐标轴连续3个X坐标对应的Y轴值：a b c, 其中b的梯度为(c-a) / 2
C = np.random.randint(0, 20, (5))
print(np.gradient(C))

D = np.random.randint(0, 20, (3,5))
print(np.gradient(D))