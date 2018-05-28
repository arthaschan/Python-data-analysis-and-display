# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     09_Pandas数据特征分析
   Author :       Zeke
   date：          2018/5/28
   Description :
-------------------------------------------------
"""

# 对一组数据的理解
# 基本统计（含排序）
# 分布/累计统计
# 数据特征（相关性/周期性等）
# 数据挖掘（形成知识）

import pandas as pd
import numpy as np

# 1/ 对数据的排序
# .sort_index(axis=0, ascending=True)方法
b = pd.DataFrame(np.arange(20).reshape(4,5),index=['c', 'a', 'd', 'b'])
print(b.sort_index())
print(b.sort_index(ascending=False))
print(b.sort_index(axis=1))

# Series.sort_values(axis=0, ascending=True)
# DataFrame.sort_values(by, axis=0, ascending=True)
# by: axis轴上的某个索引或索引列表
# NaN永远放到排序末尾
print(b.sort_values(2, ascending=False))
print(b.sort_values('a', axis=1, ascending=True))

# 2/基本统计分析函数
# .sum() .count() .mean() .median() .var() .std() .min() .max()
# 适用于Series类型
# .argmin() .argmax() 计算数据最大值最小值所在位置的索引位置(自动索引)
# .idxmin() .idxmax() 计算数据最大值最小值所在位置的索引（自定义索引）
# .describe()
print(b.describe())
print(b.describe().loc['max'])

# 3/累计统计分析
print(b.cumsum())
print(b.cumprod())
print(b.cummin())
print(b.cummax())

# 4/ 滚动窗口计算
print(b.rolling(2).sum()) # 0轴，依次计算相邻w个元素的和
print(b.rolling(2).mean())
print(b.rolling(2).var())
print(b.rolling(2).std())
print(b.rolling(2).min())
print(b.rolling(2).max())

# 5/ 数据的相关分析
# 协方差
# Pearson相关系数
# .cov() 计算协方差矩阵
# .corr() 计算相关系数矩阵，Pearson/Spearman/Kendall等系数
hprice = pd.Series([3.04, 22.93, 12.75, 22.6, 12.33],
                   index=['2008', '2009','2010','2011','2012'])
m2 = pd.Series([8.18, 18.38, 9.13, 7.82, 6.69],
               index=['2008', '2009','2010','2011','2012'])
print(hprice.corr(m2))





