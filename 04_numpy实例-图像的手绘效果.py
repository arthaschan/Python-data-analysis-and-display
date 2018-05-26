# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     numpy实例-图像的手绘效果
   Author :       Zeke
   date：          2018/5/26
   Description :
-------------------------------------------------
"""
from PIL import Image
import numpy as np

# 图像的变换1
a = np.array(Image.open(r'C:\Users\Zeke\my_code\1.jpg'))
print(a.shape, a.dtype)

b = (100 / 255) * a + 150

im = Image.fromarray(b.astype('uint8'))
im.save(r'C:\Users\Zeke\my_code\2.jpg')


# 图像的区间变换3
a = np.array(Image.open(r'C:\Users\Zeke\my_code\1.jpg').convert('L')) #RGB转换成灰度值
print(a.shape, a.dtype)

b = (100/255) * a + 150

im = Image.fromarray(b.astype('uint8'))
im.save(r'C:\Users\Zeke\my_code\3.jpg')


# 图像的手绘效果
# 几个特征
# 黑白灰色
# 边界线条较重
# 相同或相近色彩趋于白色
# 略有光源效果

a = np.array(Image.open(r'C:\Users\Zeke\my_code\1.jpg').convert('L')).astype('float') #RGB转换成灰度值
print(a.shape, a.dtype)

depth = 10
grad = np.gradient(a)
grad_x, grad_y = grad
grad_x = grad_x * depth / 100
grad_y = grad_y * depth / 100

A = np.sqrt(grad_x**2 + grad_y**2 + 1)
uni_x = grad_x / A
uni_y = grad_y / A
uni_z = 1 / A

vec_el = np.pi / 2.2
vec_az = np.pi / 4
dx = np.cos(vec_el) * np.cos(vec_az)
dy = np.cos(vec_el) * np.sin(vec_az)
dz = np.sin(vec_el)

b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
b = b.clip(0, 255)

im = Image.fromarray(b.astype('uint8'))
im.save(r'C:\Users\Zeke\my_code\4.jpg')



