# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     numpy
   Author :       arthas
   date：          2019/2/7
   Description : 高级数据练习

-------------------------------------------------
"""
import numpy as np
#如何导入数字和文本的数据集保持文本在numpy数组中完好无损？

# Solution
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Print the first 3 rows
iris[:3]
# > array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa'],
# >        [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa'],
# >        [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa']], dtype=object)
#26. 如何从1维元组数组中提取特定列？
# 从前面问题中导入的一维鸢尾属植物数据集中提取文本列的物种。
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
print(iris_1d.shape)

# Solution:
species = np.array([row[4] for row in iris_1d])
species[:5]
# > (150,)
# > array([b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
# >        b'Iris-setosa'],
# >       dtype='|S18')
#27. 如何将1维元组数组转换为2维numpy数组？
# 通过省略鸢尾属植物数据集种类的文本字段，将一维鸢尾属植物数据集转换为二维数组iris_2d。


# Solution:
# Method 1: Convert each row to a list and get the first 4 items
iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
iris_2d[:4]

# Alt Method 2: Import only the first 4 columns from source url
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[:4]
# > array([[ 5.1,  3.5,  1.4,  0.2],
# >        [ 4.9,  3. ,  1.4,  0.2],
# >        [ 4.7,  3.2,  1.3,  0.2],
# >        [ 4.6,  3.1,  1.5,  0.2]])
#28. 如何计算numpy数组的均值，中位数，标准差？
# 求出鸢尾属植物萼片长度的平均值、中位数和标准差(第1列)


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

# Solution
mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mu, med, sd)
# > 5.84333333333 5.8 0.825301291785
#29. 如何规范化数组，使数组的值正好介于0和1之间？
# 创建一种标准化形式的鸢尾属植物间隔长度，其值正好介于0和1之间，这样最小值为0，最大值为1。

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

# Solution
Smax, Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin)/(Smax - Smin)
# or
S = (sepallength - Smin)/sepallength.ptp()  # Thanks, David Ojeda!
print(S)
# > [ 0.222  0.167  0.111  0.083  0.194  0.306  0.083  0.194  0.028  0.167
# >   0.306  0.139  0.139  0.     0.417  0.389  0.306  0.222  0.389  0.222
# >   0.306  0.222  0.083  0.222  0.139  0.194  0.194  0.25   0.25   0.111
# >   0.139  0.306  0.25   0.333  0.167  0.194  0.333  0.167  0.028  0.222
# >   0.194  0.056  0.028  0.194  0.222  0.139  0.222  0.083  0.278  0.194
# >   0.75   0.583  0.722  0.333  0.611  0.389  0.556  0.167  0.639  0.25
# >   0.194  0.444  0.472  0.5    0.361  0.667  0.361  0.417  0.528  0.361
# >   0.444  0.5    0.556  0.5    0.583  0.639  0.694  0.667  0.472  0.389
# >   0.333  0.333  0.417  0.472  0.306  0.472  0.667  0.556  0.361  0.333
# >   0.333  0.5    0.417  0.194  0.361  0.389  0.389  0.528  0.222  0.389
# >   0.556  0.417  0.778  0.556  0.611  0.917  0.167  0.833  0.667  0.806
# >   0.611  0.583  0.694  0.389  0.417  0.583  0.611  0.944  0.944  0.472
# >   0.722  0.361  0.944  0.556  0.667  0.806  0.528  0.5    0.583  0.806
# >   0.861  1.     0.583  0.556  0.5    0.944  0.556  0.583  0.472  0.722
# >   0.667  0.722  0.417  0.694  0.667  0.667  0.556  0.611  0.528  0.444]
#30. 如何计算Softmax得分？
# 计算sepallength的softmax分数。

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.array([float(row[0]) for row in iris])

# Solution
def softmax(x):
    """Compute softmax values for each sets of scores in x.
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print(softmax(sepallength))
# > [ 0.002  0.002  0.001  0.001  0.002  0.003  0.001  0.002  0.001  0.002
# >   0.003  0.002  0.002  0.001  0.004  0.004  0.003  0.002  0.004  0.002
# >   0.003  0.002  0.001  0.002  0.002  0.002  0.002  0.002  0.002  0.001
# >   0.002  0.003  0.002  0.003  0.002  0.002  0.003  0.002  0.001  0.002
# >   0.002  0.001  0.001  0.002  0.002  0.002  0.002  0.001  0.003  0.002
# >   0.015  0.008  0.013  0.003  0.009  0.004  0.007  0.002  0.01   0.002
# >   0.002  0.005  0.005  0.006  0.004  0.011  0.004  0.004  0.007  0.004
# >   0.005  0.006  0.007  0.006  0.008  0.01   0.012  0.011  0.005  0.004
# >   0.003  0.003  0.004  0.005  0.003  0.005  0.011  0.007  0.004  0.003
# >   0.003  0.006  0.004  0.002  0.004  0.004  0.004  0.007  0.002  0.004
# >   0.007  0.004  0.016  0.007  0.009  0.027  0.002  0.02   0.011  0.018
# >   0.009  0.008  0.012  0.004  0.004  0.008  0.009  0.03   0.03   0.005
# >   0.013  0.004  0.03   0.007  0.011  0.018  0.007  0.006  0.008  0.018
# >   0.022  0.037  0.008  0.007  0.006  0.03   0.007  0.008  0.005  0.013
# >   0.011  0.013  0.004  0.012  0.011  0.011  0.007  0.009  0.007  0.005]
#31. 如何找到numpy数组的百分位数？
# 找到鸢尾属植物数据集的第5和第95百分位数
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

# Solution
np.percentile(sepallength, q=[5, 95])
# > array([ 4.6  ,  7.255])
#32. 如何在数组中的随机位置插入值？
#在iris_2d数据集中的20个随机位置插入np.nan值

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

# Method 1
i, j = np.where(iris_2d)

# i, j contain the row numbers and column numbers of 600 elements of iris_x
np.random.seed(100)
iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan

# Method 2
np.random.seed(100)
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Print first 10 rows
print(iris_2d[:10])
# > [[b'5.1' b'3.5' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.9' b'3.0' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
# >  [b'4.6' b'3.1' b'1.5' b'0.2' b'Iris-setosa']
# >  [b'5.0' b'3.6' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'5.4' b'3.9' b'1.7' b'0.4' b'Iris-setosa']
# >  [b'4.6' b'3.4' b'1.4' b'0.3' b'Iris-setosa']
# >  [b'5.0' b'3.4' b'1.5' b'0.2' b'Iris-setosa']
# >  [b'4.4' nan b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']]
#33. 如何在numpy数组中找到缺失值的位置？
# 在iris_2d的sepallength中查找缺失值的数量和位置（第1列）

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution
print("Number of missing values: \n", np.isnan(iris_2d[:, 0]).sum())
print("Position of missing values: \n", np.where(np.isnan(iris_2d[:, 0])))
# > Number of missing values:
# >  5
# > Position of missing values:
# >  (array([ 39,  88,  99, 130, 147]),)
#34. 如何根据两个或多个条件过滤numpy数组？
# 过滤具有petallength（第3列）> 1.5 和 sepallength（第1列）< 5.0 的iris_2d行

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# Solution
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
iris_2d[condition]
# > array([[ 4.8,  3.4,  1.6,  0.2],
# >        [ 4.8,  3.4,  1.9,  0.2],
# >        [ 4.7,  3.2,  1.6,  0.2],
# >        [ 4.8,  3.1,  1.6,  0.2],
# >        [ 4.9,  2.4,  3.3,  1. ],
# >        [ 4.9,  2.5,  4.5,  1.7]])
#35. 如何从numpy数组中删除包含缺失值的行？
# 选择没有任何nan值的iris_2d行。
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution
# No direct numpy function for this.
# Method 1:
any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
iris_2d[any_nan_in_row][:5]

# Method 2: (By Rong)
iris_2d[np.sum(np.isnan(iris_2d), axis = 1) == 0][:5]
# > array([[ 4.9,  3. ,  1.4,  0.2],
# >        [ 4.7,  3.2,  1.3,  0.2],
# >        [ 4.6,  3.1,  1.5,  0.2],
# >        [ 5. ,  3.6,  1.4,  0.2],
# >        [ 5.4,  3.9,  1.7,  0.4]])

#38. 如何在numpy数组中用0替换所有缺失值？
# 在numpy数组中将所有出现的nan替换为0

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution
iris_2d[np.isnan(iris_2d)] = 0
iris_2d[:4]
# > array([[ 5.1,  3.5,  1.4,  0. ],
# >        [ 4.9,  3. ,  1.4,  0.2],
# >        [ 4.7,  3.2,  1.3,  0.2],
# >        [ 4.6,  3.1,  1.5,  0.2]])
#39. 如何在numpy数组中查找唯一值的计数？
# 找出鸢尾属植物物种中的独特值和独特值的数量

# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Solution
# Extract the species column as an array
species = np.array([row.tolist()[4] for row in iris])

# Get the unique values and the counts
np.unique(species, return_counts=True)
# > (array([b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica'],
# >        dtype='|S15'), array([50, 50, 50]))
#40. 如何将数字转换为分类（文本）数组？
# 将iris_2d的花瓣长度（第3列）加入以形成文本数组，这样如果花瓣长度为：

#Less than 3 --> 'small'
#3-5 --> 'medium'
#'>=5 --> 'large'


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Bin petallength
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])

# Map it to respective category
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]

# View
petal_length_cat[:4]
# > ['small', 'small', 'small', 'small']

#44. 如何按列对2D数组进行排序
# 根据sepallength列对虹膜数据集进行排序。

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
# Sort by column position 0: SepalLength
print(iris[iris[:,0].argsort()][:20])
# > [[b'4.3' b'3.0' b'1.1' b'0.1' b'Iris-setosa']
# >  [b'4.4' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
# >  [b'4.4' b'3.0' b'1.3' b'0.2' b'Iris-setosa']
# >  [b'4.4' b'2.9' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.5' b'2.3' b'1.3' b'0.3' b'Iris-setosa']
# >  [b'4.6' b'3.6' b'1.0' b'0.2' b'Iris-setosa']
# >  [b'4.6' b'3.1' b'1.5' b'0.2' b'Iris-setosa']
# >  [b'4.6' b'3.4' b'1.4' b'0.3' b'Iris-setosa']
# >  [b'4.6' b'3.2' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
# >  [b'4.7' b'3.2' b'1.6' b'0.2' b'Iris-setosa']
# >  [b'4.8' b'3.0' b'1.4' b'0.1' b'Iris-setosa']
# >  [b'4.8' b'3.0' b'1.4' b'0.3' b'Iris-setosa']
# >  [b'4.8' b'3.4' b'1.9' b'0.2' b'Iris-setosa']
# >  [b'4.8' b'3.4' b'1.6' b'0.2' b'Iris-setosa']
# >  [b'4.8' b'3.1' b'1.6' b'0.2' b'Iris-setosa']
# >  [b'4.9' b'2.4' b'3.3' b'1.0' b'Iris-versicolor']
# >  [b'4.9' b'2.5' b'4.5' b'1.7' b'Iris-virginica']
# >  [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']
# >  [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']]
#45. 如何在numpy数组中找到最常见的值？
# 在鸢尾属植物数据集中找到最常见的花瓣长度值（第3列）。
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Solution:
vals, counts = np.unique(iris[:, 2], return_counts=True)
print(vals[np.argmax(counts)])



#如何找到第一次出现的值大于给定值的位置？
#在虹膜数据集的petalwidth第4列中查找第一次出现的值大于1.0的位置。

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution: (edit: changed argmax to argwhere. Thanks Rong!)
np.argwhere(iris[:, 3].astype(float) > 1.0)[0]
# > 50
#47. 如何将大于给定值的所有值替换为给定的截止值？
# 从数组a中，替换所有大于30到30和小于10到10的值。

# Input
np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1,50, 20)

# Solution 1: Using np.clip
np.clip(a, a_min=10, a_max=30)

# Solution 2: Using np.where
print(np.where(a < 10, 10, np.where(a > 30, 30, a)))
# > [ 27.63  14.64  21.8   30.    10.    10.    30.    30.    10.    29.18  30.
# >   11.25  10.08  10.    11.77  30.    30.    10.    30.    14.43]
#48. 如何从numpy数组中获取最大n值的位置？
# 获取给定数组a中前5个最大值的位置。

# Input
np.random.seed(100)
a = np.random.uniform(1,50, 20)

# Solution:
print(a.argsort())
# > [18 7 3 10 15]

# Solution 2:
np.argpartition(-a, 5)[:5]
# > [15 10  3  7 18]

# Below methods will get you the values.
# Method 1:
a[a.argsort()][-5:]

# Method 2:
np.sort(a)[-5:]

# Method 3:
np.partition(a, kth=-5)[-5:]

# Method 4:
a[np.argpartition(-a, 5)][:5]
#49. 如何计算数组中所有可能值的行数？
# 按行计算唯一值的计数。

np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))
arr
# > array([[ 9,  9,  4,  8,  8,  1,  5,  3,  6,  3],
# >        [ 3,  3,  2,  1,  9,  5,  1, 10,  7,  3],
# >        [ 5,  2,  6,  4,  5,  5,  4,  8,  2,  2],
# >        [ 8,  8,  1,  3, 10, 10,  4,  3,  6,  9],
# >        [ 2,  1,  8,  7,  3,  1,  9,  3,  6,  2],
# >        [ 9,  2,  6,  5,  3,  9,  4,  6,  1, 10]])
# Solution
def counts_of_all_values_rowwise(arr2d):
    # Unique values and its counts row wise
    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]

    # Counts of all values row wise
    return([[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array])

# Print
print(np.arange(1,11))
counts_of_all_values_rowwise(arr)
# > [ 1  2  3  4  5  6  7  8  9 10]

# > [[1, 0, 2, 1, 1, 1, 0, 2, 2, 0],
# >  [2, 1, 3, 0, 1, 0, 1, 0, 1, 1],
# >  [0, 3, 0, 2, 3, 1, 0, 1, 0, 0],
# >  [1, 0, 2, 1, 0, 1, 0, 2, 1, 2],
# >  [2, 2, 2, 0, 0, 1, 1, 1, 1, 0],
# >  [1, 1, 1, 1, 1, 2, 0, 0, 2, 1]]
# Example 2:
arr = np.array([np.array(list('bill clinton')), np.array(list('narendramodi')), np.array(list('jjayalalitha'))])
print(np.unique(arr))
counts_of_all_values_rowwise(arr)
# > [' ' 'a' 'b' 'c' 'd' 'e' 'h' 'i' 'j' 'l' 'm' 'n' 'o' 'r' 't' 'y']

# > [[1, 0, 1, 1, 0, 0, 0, 2, 0, 3, 0, 2, 1, 0, 1, 0],
# >  [0, 2, 0, 0, 2, 1, 0, 1, 0, 0, 1, 2, 1, 2, 0, 0],
# >  [0, 4, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1]]
#50. 如何将数组转换为平面一维数组？
#
# 将array_of_arrays转换为扁平线性1d数组。


 # **给定：**
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)

array_of_arrays = np.array([arr1, arr2, arr3])
print('array_of_arrays: ', array_of_arrays)

# Solution 1
arr_2d = np.array([a for arr in array_of_arrays for a in arr])

# Solution 2:
arr_2d = np.concatenate(array_of_arrays)
print(arr_2d)
# > array_of_arrays:  [array([0, 1, 2]) array([3, 4, 5, 6]) array([7, 8, 9])]
# > [0 1 2 3 4 5 6 7 8 9]
#52. 如何创建按分类变量分组的行号？
# 创建按分类变量分组的行号。使用以下来自鸢尾属植物物种的样本作为输入。
# **给定：**
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
np.random.seed(100)
species_small = np.sort(np.random.choice(species, size=20))
species_small
# > array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
# >        'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica'],
# >       dtype='<U15')
print([i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])])
[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5]
#53. 如何根据给定的分类变量创建组ID？
# 根据给定的分类变量创建组ID。使用以下来自鸢尾属植物物种的样本作为输入。

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
np.random.seed(100)
species_small = np.sort(np.random.choice(species, size=20))
species_small
# > array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
# >        'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica'],
# >       dtype='<U15')
# Solution:
output = [np.argwhere(np.unique(species_small) == s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]]

# Solution: For Loop version
output = []
uniqs = np.unique(species_small)

for val in uniqs:  # uniq values in group
    for s in species_small[species_small==val]:  # each element in group
        groupid = np.argwhere(uniqs == s).tolist()[0][0]  # groupid
        output.append(groupid)

print(output)
# > [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
#54. 如何使用numpy对数组中的项进行排名？
# 为给定的数字数组a创建排名。
np.random.seed(10)
a = np.random.randint(20, size=10)
print('Array: ', a)

# Solution
print(a.argsort().argsort())
print('Array: ', a)
# > Array:  [ 9  4 15  0 17 16 17  8  9  0]
# > [4 2 6 0 8 7 9 3 5 1]
# > Array:  [ 9  4 15  0 17 16 17  8  9  0]
#55. 如何使用numpy对多维数组中的项进行排名？
# 创建与给定数字数组a相同形状的排名数组。

# **给定：**
np.random.seed(10)
a = np.random.randint(20, size=[2,5])
print(a)

# Solution
print(a.ravel().argsort().argsort().reshape(a.shape))
# > [[ 9  4 15  0 17]
# >  [16 17  8  9  0]]
# > [[4 2 6 0 8]
# >  [7 9 3 5 1]]
#56. 如何在二维numpy数组的每一行中找到最大值？
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a

# Solution 1
np.amax(a, axis=1)

# Solution 2
np.apply_along_axis(np.max, arr=a, axis=1)
# > array([9, 8, 6, 3, 9])
#57. 如何计算二维numpy数组每行的最小值？
# 为给定的二维numpy数组计算每行的最小值。
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a

# Solution
np.apply_along_axis(lambda x: np.min(x)/np.max(x), arr=a, axis=1)
# > array([ 0.44444444,  0.125     ,  0.5       ,  1.        ,  0.11111111])
#58. 如何在numpy数组中找到重复的记录？
# 在给定的numpy数组中找到重复的条目(第二次出现以后)，并将它们标记为True。第一次出现应该是False的。
np.random.seed(100)
a = np.random.randint(0, 5, 10)

## Solution
# There is no direct function to do this as of 1.13.3

# Create an all True array
out = np.full(a.shape[0], True)

# Find the index positions of unique elements
unique_positions = np.unique(a, return_index=True)[1]

# Mark those positions as False
out[unique_positions] = False

print(out)
# > [False  True False  True False False  True  True  True  True]
#59. 如何找出数字的分组均值？
# 在二维数字数组中查找按分类列分组的数值列的平均值

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')


# Solution
# No direct way to implement this. Just a version of a workaround.
numeric_column = iris[:, 1].astype('float')  # sepalwidth
grouping_column = iris[:, 4]  # species

# List comprehension version
[[group_val, numeric_column[grouping_column==group_val].mean()] for group_val in np.unique(grouping_column)]

# For Loop version
output = []
for group_val in np.unique(grouping_column):
    output.append([group_val, numeric_column[grouping_column==group_val].mean()])

output
# > [[b'Iris-setosa', 3.418],
# >  [b'Iris-versicolor', 2.770],
# >  [b'Iris-virginica', 2.974]]
#60. 如何将PIL图像转换为numpy数组？
# 从以下URL导入图像并将其转换为numpy数组。


from io import BytesIO
from PIL import Image
import PIL, requests

# Import image from URL
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
response = requests.get(URL)

# Read it as Image
I = Image.open(BytesIO(response.content))

# Optionally resize
I = I.resize([150,150])

# Convert to numpy array
arr = np.asarray(I)

# Optionaly Convert it back to an image and show
im = PIL.Image.fromarray(np.uint8(arr))
Image.Image.show(im)
#61. 如何删除numpy数组中所有缺少的值？
# 从一维numpy数组中删除所有NaN值

a = np.array([1,2,3,np.nan,5,6,7,np.nan])
a[~np.isnan(a)]
# > array([ 1.,  2.,  3.,  5.,  6.,  7.])
#62. 如何计算两个数组之间的欧氏距离？
# 计算两个数组a和数组b之间的欧氏距离。
# Input
a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])

# Solution
dist = np.linalg.norm(a-b)
dist
# > 6.7082039324993694
#63. 如何在一维数组中找到所有的局部极大值(或峰值)？
# 找到一个一维数字数组a中的所有峰值。峰顶是两边被较小数值包围的点。
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
doublediff = np.diff(np.sign(np.diff(a)))
peak_locations = np.where(doublediff == -2)[0] + 1
peak_locations
# > array([2, 5])
#64. 如何从二维数组中减去一维数组，其中一维数组的每一项从各自的行中减去？
# 从2d数组a_2d中减去一维数组b_1D，使得b_1D的每一项从a_2d的相应行中减去。
# Input
a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,2,3])

# Solution
print(a_2d - b_1d[:,None])
# > [[2 2 2]
# >  [2 2 2]
# >  [2 2 2]]
#65. 如何查找数组中项的第n次重复索引？
# 找出x中数字1的第5次重复的索引。

x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
n = 5

# Solution 1: List comprehension
[i for i, v in enumerate(x) if v == 1][n-1]

# Solution 2: Numpy version
np.where(x == 1)[0][n-1]
# > 8
#66. 如何将numpy的datetime 64对象转换为datetime的datetime对象？
# ：将numpy的datetime64对象转换为datetime的datetime对象

# **给定：** a numpy datetime64 object
dt64 = np.datetime64('2018-02-25 22:10:10')

# Solution
from datetime import datetime
dt64.tolist()

# or

dt64.astype(datetime)
# > datetime.datetime(2018, 2, 25, 22, 10, 10)
#67. 如何计算numpy数组的移动平均值？
# 对于给定的一维数组，计算窗口大小为3的移动平均值。


# Solution
# Source: https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

np.random.seed(100)
Z = np.random.randint(10, size=10)
print('array: ', Z)
# Method 1
moving_average(Z, n=3).round(2)

# Method 2:  # Thanks AlanLRH!
# np.ones(3)/3 gives equal weights. Use np.ones(4)/4 for window size 4.
np.convolve(Z, np.ones(3)/3, mode='valid') 


# > array:  [8 8 3 7 7 0 4 2 5 2]
# > moving average:  [ 6.33  6.    5.67  4.67  3.67  2.    3.67  3.  ]
#68. 如何在给定起始点、长度和步骤的情况下创建一个numpy数组序列？
# 创建长度为10的numpy数组，从5开始，在连续的数字之间的步长为3。
length = 10
start = 5
step = 3

def seq(start, length, step):
    end = start + (step*length)
    return np.arange(start, end, step)

seq(start, length, step)
# > array([ 5,  8, 11, 14, 17, 20, 23, 26, 29, 32])
#69. 如何填写不规则系列的numpy日期中的缺失日期？
# 给定一系列不连续的日期序列。填写缺失的日期，使其成为连续的日期序列。
# Input
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
print(dates)

# Solution ---------------
filled_in = np.array([np.arange(date, (date+d)) for date, d in zip(dates, np.diff(dates))]).reshape(-1)

# add the last day
output = np.hstack([filled_in, dates[-1]])
output

# For loop version -------
out = []
for date, d in zip(dates, np.diff(dates)):
    out.append(np.arange(date, (date+d)))

filled_in = np.array(out).reshape(-1)

# add the last day
output = np.hstack([filled_in, dates[-1]])
output
# > ['2018-02-01' '2018-02-03' '2018-02-05' '2018-02-07' '2018-02-09'
# >  '2018-02-11' '2018-02-13' '2018-02-15' '2018-02-17' '2018-02-19'
# >  '2018-02-21' '2018-02-23']

# > array(['2018-02-01', '2018-02-02', '2018-02-03', '2018-02-04',
# >        '2018-02-05', '2018-02-06', '2018-02-07', '2018-02-08',
# >        '2018-02-09', '2018-02-10', '2018-02-11', '2018-02-12',
# >        '2018-02-13', '2018-02-14', '2018-02-15', '2018-02-16',
# >        '2018-02-17', '2018-02-18', '2018-02-19', '2018-02-20',
# >        '2018-02-21', '2018-02-22', '2018-02-23'], dtype='datetime64[D]')
#70. 如何从给定的一维数组创建步长？
# 从给定的一维数组arr中，利用步进生成一个二维矩阵，窗口长度为4，步距为2，类似于 [[0,1,2,3], [2,3,4,5], [4,5,6,7]..]
def gen_strides(a, stride_len=5, window_len=5):
    n_strides = ((a.size-window_len)//stride_len) + 1
    # return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])
    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])

print(gen_strides(np.arange(15), stride_len=2, window_len=4))
# > [[ 0  1  2  3]
# >  [ 2  3  4  5]
# >  [ 4  5  6  7]
# >  [ 6  7  8  9]
# >  [ 8  9 10 11]
# >  [10 11 12 13]]


#https://www.machinelearningplus.com/python/101-numpy-exercises-python/
#https://www.numpy.org.cn/article/advanced/numpy_exercises_for_data_analysis.html