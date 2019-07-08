
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:49:14 2019

@author: NewDreamstyle
"""
import numpy as np

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
threshold = 0.17
tree = []  #【（最优切分点，左均值，右均值，loss）】
loss = 1

# 寻找最优切分点并记录
num = 0
while loss > threshold and num < 10:
    split = []  #【切分点】
    final = []
    mean  = []
    for i in range(len(x_train)-1):
        split.append((x_train[i]+x_train[i+1])/2)
    for i in split:
        left_index = np.array([k for k in range(len(x_train)) if x_train[k] <= i])
        left = np.array([y_train[k] for k in left_index])
        a = sum([(i-left.mean())**2 for i in left])

        right_index = np.array([k for k in range(len(x_train)) if x_train[k] > i])
        right = np.array([y_train[k] for k in right_index])
        b = sum([(i-right.mean())**2 for i in right])
        final.append(a+b)
        mean.append((left.mean(),right.mean()))
    argmin_index = np.array(final).argmin()
    loss = final[argmin_index]
    tree.append((split[argmin_index],mean[argmin_index][0],mean[argmin_index][1],loss))
    # 计算残差
    for i in range(len(y_train)):
        if x_train[i] < tree[-1][0]:
            y_train[i] = y_train[i] - tree[-1][1] 
        else:
            y_train[i] = y_train[i] - tree[-1][2] 
    num += 1