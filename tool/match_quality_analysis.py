# !/usr/bin/env python
# -*- coding:utf-8 -*- 


import numpy as np
import matplotlib.pyplot as plt 


inlier_ratio = np.loadtxt('bin/match_inlier_ratio_0.5')
# print(inlier_ratio)
# print(len(inlier_ratio[inlier_ratio<0.1]))
plt.figure(figsize=(10,8)) 
plt.hist(inlier_ratio) 
# plt.title('')
plt.xlabel('inlier_ratio') 
plt.ylabel('ratio') 
plt.show()