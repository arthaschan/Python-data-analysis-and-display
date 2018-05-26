# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     numpy
   Author :       Zeke
   date：          2018/5/26
   Description :对numpy的学习
   1 有一个强大的N维数组对象ndarray
   2 广播功能函数
   3 整合C/C++？Fortran代码的工具
   4 线性代数.傅里叶变换.随机数生成等功能
   是SciPy/Pandas等数据处理与科学计算的基础
-------------------------------------------------
"""
# ndarray是一个多维数组对象
# 包含实际的数据与描述这些数据的元数据（数据维度与数据类型等）
# ndarray在程序中的别名是array
import numpy as np

a = np.array([[0,1,2,3,4],
             [9,8,7,6,5]])

print(a)
print(a.ndim)  # 秩，即轴的数量或维度的数量
print(a.shape) # ndarray对象的尺度，对于矩阵，n行m列
print(a.size)  # ndarray对象元素的个数，相当于a.shape中n*m的值
print(a.dtype) # ndarray对象的元素类型
print(a.itemsize)  #ndarray对象中每个元素的大小，以字节为单位

# nparray的创建
# (1)
# x = np.array(list/tuple)
# x = np.array(list/tuple,dtype=np.float32)
# (2)
print('\nnparray的创建')
print(np.arange(10))
print(np.ones((3,6)))
print(np.zeros((3,6),dtype=np.int32))
print(np.eye(5))
print('')
print(np.ones((2,3,4)))
print(np.ones((2,3,4)).shape)
print(np.full((2,3,4),9))
# (3)
print('')
print(np.ones_like(a))
print(np.zeros_like(a))
print(np.full_like(a,9)) #生成与a同样维度的数组，元素全是9
# (4)
print('')
print(np.linspace(0,10,5)) # 包括10
print(np.linspace(0,10,5,endpoint=False)) # 不包括10
print(np.concatenate((np.arange(10),np.linspace(0,10,5)))) #合并两个数组


# nparray的维度变换
print('\nnparray的维度变换')
b = np.ones((2,3,4),dtype=np.int32)
print(b.reshape((3,8))) #reshape不改变原数组
print(b.resize((3,8))) #resize改变原数组
print(b.flatten()) # 数组降维成1维


# nparray的类型变换
print('\nnparray的类型度变换')
c= np.ones((2,3,4),dtype=np.int)
print(c)
new_c = c.astype(np.float) # 一定会生成一个数组，可以通过它对数组进行拷贝
print(new_c)


# nparray数组向列表的转换
print('\nnparray数组向列表的转换')
d = np.full((2,3,4),25,dtype=np.int32)
print(d)
print(d.tolist())

# nparray数组的操作
print('\nnparray数组的操作')
e = np.linspace(1,10,10)
print(e[1:4:1]) # 起始编号：终止编号（不含）：步长

print('')
f = np.arange(24).reshape((2,3,4))
print(f[1,2,3])
print(f[-1,-2,-3])

print(f[:,1:,-3])
print(f[:,1:3,:]) # 切片
print(f[:,:,::2])



# nparray数组的运算
# 一元函数(均返回新数组)
print('\nnparray数组的运算')
g = np.arange(1,6) / 5
print(np.abs(g))
print(np.sqrt(g))
print(np.square(g))
print(np.log(g))
print(np.log10(g))
print(np.log2(g))
print(np.ceil(g))
print(np.floor(g))

print(np.rint(g)) # 将数组个元素四舍五入
print(np.modf(g)) # 将数组个元素的小数和整数部分以两个独立数组形式返回
print(np.cos(g))
print(np.exp(g))
print(np.sign(g)) #计算个元素的符号值


# 二元函数(均返回新数组)
# + - * / **
# np.maximum(x,y)  np.fmax()
# np.minimum(x,y)  np.fmin()
# np.mod(x,y) # 元素级别的模运算
# np.copysign(x,y) # 将数组y中个元素的符号赋值给数组x对应元素
# > < >= <= == !=


