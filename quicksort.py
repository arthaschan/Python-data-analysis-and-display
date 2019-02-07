# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     numpy
   Author :       arthas
   date：          2019/2/7
   Description :快速排序算法。
   递归，每一次，先找中间位置的那个索引，pivot.比中间数小的在左边，比中间数大的在右边。

-------------------------------------------------
"""
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))