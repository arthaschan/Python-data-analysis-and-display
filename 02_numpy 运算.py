# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     numpy
   Author :       arthas
   date：          2019/2/7
   Description :  矩阵的运算
   x+y  np.add(x,y)
   x-y  np.subtract(x,y)
   x*y  np.multiply(x,y)  元素直接相乘
   x/y np.divide(x,y)
    v.dot(w)  np.dot(v,w)  点乘，矩阵的真正的乘法。

    transpose  转置
-------------------------------------------------
"""
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))


v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))


x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"  求所有元素的和
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"   上下相加，同一列相加
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"     左右相加   同一行相加


A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]) )
# AX=b。求X
x = np.linalg.solve(A,b)