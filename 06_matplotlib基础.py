# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     06_matplotlib基础
   Author :       Zeke
   date：          2018/5/26
   Description :
-------------------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np


# 饼图的绘制
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0) # 指出那一块需要突出来，这里是Hogs

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
plt.axis('equal')
plt.show()


# 直方图的绘制
np.random.seed(123)
mu, sigma = 100, 20 #均值和标准差
a = np.random.normal(mu, sigma, size=100)

plt.hist(a, 40, normed=1, histtype='stepfilled', facecolor='b', alpha=0.75)
plt.title('Histogram')
plt.show()






















