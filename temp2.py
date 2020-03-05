from sympy import var
r, t, d = var('rate time short_life')
d = r*t+5*t
d

r = 80
t = 2
d         # We haven't changed d, only r and t

d = r*t
d

c, d = var('c d')
c
d

def ctimesd():
    """
    This function returns whatever c is times whatever d is.
    """
    return c*d

print(type(ctimesd()))

c = 2
c

print(ctimesd())









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