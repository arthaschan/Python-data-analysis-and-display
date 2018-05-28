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


# 极坐标图的绘制
N = 20
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, projection='polar')
bars = ax.bar(theta, radii, width=width, bottom=0)

for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.viridis(r / 10))
        bar.set_alpha(0.5)
plt.show()


# 散点图的绘制(面向对象的方法绘制)
fig, ax = plt.subplots()
ax.plot(10*np.random.rand(100), 10*np.random.rand(100), 'o')
ax.set_title('Simple Scatter')
plt.show()




























