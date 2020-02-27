# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:17:13 2015

@author: Eddy_zheng
"""

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()












'''

from sympy import *
import math
import matplotlib.pyplot as plt
import numpy as np

x = symbols("x") # 符号x，自变量
#y = -pow(10,-11)*pow(x,6) + pow(10,-8)*pow(x,5) - 4*pow(10,-6)*pow(x,4) + 0.0006*pow(x,3) - 0.0428*pow(x,2) + 1.7561*x + 16.528 #公式
y = 1 / (1 + exp((x-5)*-1))
dify = diff(y,x) #求导
print(dify) #打印导数

#给定x值，求对应导数的值
x =[]
y_value= []
dify_value =[]
for i in np.arange(0,10,0.1):
    x.append(i)
    #print(x)
    y_value.append(float(y.subs('x',i)))
    dify_value.append(float(dify.subs('x',i)))
    #print(dify_value)

#x = np.array(x)
#dify_value = np.array(dify_value)
plt.figure()
plt.plot(x, y_value)
plt.plot(x, dify_value)
plt.show()
'''