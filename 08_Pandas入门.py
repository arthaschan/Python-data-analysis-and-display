# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     08_Pandas入门
   Author :       Zeke
   date：          2018/5/28
   Description :
-------------------------------------------------
"""

import pandas as pd

# Series类型------一维的带标签的数组
# 从列表创建
a = pd.Series([9,8,7,6]) #自动生成索引
b = pd.Series([9,8,7,6],index=['a','b','c','d']) #自定义索引
# 从标量创建
s = pd.Series(25, index=['a','b','c']) #不能省略index
# 从字典创建
d = pd.Series({'a':9,'b':8,'c':7})
e = pd.Series({'a':9,'b':8,'c':7},index=['c','a','b','d']) #d对应Nan
# 从ndarray创建
import numpy as np
n = pd.Series(np.arange(5))
m = pd.Series(np.arange(5), index=np.arange(9,4,-1))


# 操作
print(b.index)
print(b.values)
print(b['b'])
print(b[1]) #两种索引均可，但不可以混合使用
print(b[:3])
print(b[b > b.median()])
print(np.exp(b))

print('c' in b) # True
print(0 in b) # False
print(b.get('f', 100)) # 100
print(b+s) # 包含所有索引的并集
b.name = 'Series对象'
b.index.name = '索引列'
print(b)

b['a'] = 15
b['b', 'c'] = 20 #随时修改并即使生效

# DataFrame类型 ------ 共用相同索引的一组列组成
# index（axis=0），column（axis=1）

# 创建
x = pd.DataFrame(np.arange(10).reshape(2, 5))
yt = {'one':pd.Series([1,2,3],index=['a','b','c']),
      'Two':pd.Series([9,8,7,6],index=['a','b','c','d'])}
y = pd.DataFrame(yt)
print(pd.DataFrame(yt,index=['b','c','d'],columns=['two','three']))

# 使用列表类型的字典创建
g1 = {'one':[1,2,3,4],'two':[9,8,7,6]}
g = pd.DataFrame(g1,index=['a','b','c','d'])


## 重新索引
# .reindex()能够改变或重排Series和dataFrame索引
# .reindex(index=None,columns=None,...)的参数
# fill_value: 重新索引中，用于填充缺失位置的值
# method: 填充方法，ffill为当前值向前填充，bfill为向后填充
# limit: 最大填充量
# copy: 默认True，生成新的对象，False，新旧相等不复制
print(y.reindex(index=['d','c','b','a']))
print(y.reindex(columns=['Two','one']))

# 索引类型
# .index与.column
# Index对象是不可修改类型

# 使用.drop()能够删除Series和DataFrame制定行或列索引
a1 = pd.Series([9,8,7,6],index=['a','b','c','d'])
print(a1.drop(['b','c'])) # 默认删除行，如果想删除列，需要加上axis=1


# 算数运算法则
# + - * /
# add sub mul div  #好处是可以增加额外参数
z1 = pd.DataFrame(np.arange(12).reshape(3,4))
z2 = pd.DataFrame(np.arange(20).reshape(4,5))
print(z1+z2)
print(z2.add(z1,fill_value=0)) # fill_value参数替代Nan，替代后参与运算
print(z1*z2)
print(z2.mul(z1,fill_value=0))

# 不同维度间为广播运算

# 比较运算，生成布尔值
# 比较运算不进行补齐，可以广播运算
# > >= < <= == !=



