# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     numpy
   Author :       arthas
   date：          2019/2/7
   Description : 数据分析的练习

-------------------------------------------------
"""
#1、导入numpy作为np，并查看版本
import numpy as np
print(np.__version__)
# > 1.13.3

#2、如何创建一维数组？
arr = np.arange(10)
# > array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#3. 如何创建一个布尔数组？
np.full((3, 3), True, dtype=bool)
# > array([[ True,  True,  True],
# >        [ True,  True,  True],
# >        [ True,  True,  True]], dtype=bool)

# Alternate method:
np.ones((3,3), dtype=bool)

#4. 如何从一维数组中提取满足指定条件的元素？
#问题：从 arr 中提取所有的奇数
# Input
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Solution
arr[arr % 2 == 1]
# > array([1, 3, 5, 7, 9])

#5. 如何用numpy数组中的另一个值替换满足条件的元素项？
#将arr中的所有奇数替换为-1。
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# >  array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
arr[arr % 2 == 1] = -1
# > array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])

#6. 如何在不影响原始数组的情况下替换满足条件的元素项？
 #将arr中的所有奇数替换为-1，而不改变arr。

arr = np.arange(10)
out = np.where(arr % 2 == 1, -1, arr)
print(arr)
# > [0 1 2 3 4 5 6 7 8 9]
#rray([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])

#7. 如何改变数组的形状？
# #将一维数组转换为2行的2维数组

arr = np.arange(10)
arr.reshape(2, -1)  # Setting to -1 automatically decides the number of cols
# > array([[0, 1, 2, 3, 4],
# >        [5, 6, 7, 8, 9]])

#8. 如何垂直叠加两个数组？
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
# Answers
# Method 1:
np.concatenate([a, b], axis=0)
# Method 2:
np.vstack([a, b])
# Method 3:
np.r_[a, b]
# > array([[0, 1, 2, 3, 4],
# >        [5, 6, 7, 8, 9],
# >        [1, 1, 1, 1, 1],
# >        [1, 1, 1, 1, 1]])

#9. 如何水平叠加两个数组？
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
# Answers
# Method 1:
np.concatenate([a, b], axis=1)
# Method 2:
np.hstack([a, b])
# Method 3:
np.c_[a, b]
# > array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
# >        [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])

#11. 如何获取两个numpy数组之间的公共项？

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)
# > array([2, 4])

#12. 如何从一个数组中删除存在于另一个数组中的项？
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])

# From 'a' remove all of 'b'
np.setdiff1d(a,b)
# > array([1, 2, 3, 4])
#13. 如何得到两个数组元素匹配的位置？
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.where(a == b)
# > (array([1, 3, 5, 7]),)
#14. 如何从numpy数组中提取给定范围内的所有数字？
#问题：获取5到10之间的所有项目。


a = np.arange(15)

# Method 1
index = np.where((a >= 5) & (a <= 10))
a[index]

# Method 2:
index = np.where(np.logical_and(a>=5, a<=10))
a[index]
# > (array([6, 9, 10]),)

# Method 3: (thanks loganzk!)
a[(a >= 5) & (a <= 10)]

#16. 如何交换二维numpy数组中的两列？
#在数组arr中交换列1和2。

arr = np.arange(9).reshape(3,3)

# Solution
arr[:, [1,0,2]]  # 指在列上 0 1 索引互换
# > array([[1, 0, 2],
# >        [4, 3, 5],
# >        [7, 6, 8]])
#17. 如何交换二维numpy数组中的两行？
# #交换数组arr中的第1和第2行：
arr = np.arange(9).reshape(3,3)

# Solution
arr[[1,0,2], :]
# > array([[3, 4, 5],
# >        [0, 1, 2],
# >        [6, 7, 8]])

#18. 如何反转二维数组的行？
# Input
arr = np.arange(9).reshape(3,3)
# Solution
arr[::-1]
#array([[6, 7, 8],
#       [3, 4, 5],
 #      [0, 1, 2]])
#19. 如何反转二维数组的列？
arr = np.arange(9).reshape(3,3)

# Solution
arr[:, ::-1]
# > array([[2, 1, 0],
# >        [5, 4, 3],
# >        [8, 7, 6]])
#20. 如何创建包含5到10之间随机浮动的二维数组？
#问题：创建一个形状为5x3的二维数组，以包含5到10之间的随机十进制数。
 
# Input
arr = np.arange(9).reshape(3,3)

# Solution Method 1:
rand_arr = np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))
# print(rand_arr)

# Solution Method 2:
rand_arr = np.random.uniform(5,10, size=(5,3))
print(rand_arr)
# > [[ 8.50061025  9.10531502  6.85867783]
# >  [ 9.76262069  9.87717411  7.13466701]
# >  [ 7.48966403  8.33409158  6.16808631]
# >  [ 7.75010551  9.94535696  5.27373226]
# >  [ 8.0850361   5.56165518  7.31244004]]