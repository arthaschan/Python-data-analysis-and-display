# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     numpy数据存储
   Author :       Zeke
   date：          2018/5/26
   Description :   numpy数据存储知识
-------------------------------------------------
"""
import numpy as np
a = np.arange(10).reshape(2,5)


# 1/ CSV文件(只能有效存取一维和二维数组)
    # np.savetxt(frame, array, fmt='%.18e',delimiter=None)
        # frame :文件/字符串或产生器，可以是.gz或.bz2的压缩文件
        # array :存入文件的数组
        # fmt   :写入文件的格式，例如：%d %.2f %.18e
        # delimiter: 分割字符串，默认为空格
    # np.loadtxt(frame, dtype=np.float, delimiter=None, unpack=alse)

np.savetxt('a.csv', a, fmt='%d', delimiter=',')
b = np.loadtxt('a.csv', dtype=np.int, delimiter=',')


# 2/ a.tofile方法(存取时需要知道数组的维度与元素类型)

    # a.tofile(frame, sep='', format='%s')
    # np.fromfile(frame, dtype=float, count=-1, sep='')
        # count: 读入元素个数，-1表示读入整个文件
        # sep： 数据分割字符串，如果是空串，写入文件为二进制d

a.tofile('a.dat', sep=',', format='%d')
# a.tofile('a.dat', format='%d') # 不指定分隔符会生成二进制文件
b = np.fromfile('a.dat',dtype=np.int, sep=',').reshape(2,5)


# 3/ Numpy便捷的文件存储(最理想的存储方式)
    # np.save(fname, array) 或 np.savez(fname, array)
        # frame: 文件名，以.npy为扩展名，压缩扩展名为.npz
        # array: 数组变量
    # np.load(fname)
np.save('a.npy', a)
b = np.load('a.npy')




















