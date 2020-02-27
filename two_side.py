import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from sympy import *
import math

from numpy import poly1d
#from config2 import *
from mpl_toolkits.mplot3d import Axes3D

style.use('ggplot')
matplotlib.use( 'tkagg' )


class BasisParameter(object):
    '''
    define several parameter
    '''

    def __init__(self,n, cfg2):
        '''
        '''
        self.n = n
        self.cfg = cfg2
        #print(self.cfg)
    def consumer(self):
        '''
        self.u =  surplus obtained  from the n products;  linear function
        define the inflection point n=10

        '''

        if (self.n > self.cfg.INFLECTION_POINT):
            u = self.cfg.A + self.cfg.B * self.cfg.INFLECTION_POINT
        else:
            u = self.cfg.A + self.cfg.B * self.n


        #self.epsilon = theta *self.f/self.F
        return u
    def consumer_theta(self):

        theta = symbols("theta")
        F = 1/(1 + exp(-(theta-5)))
        f = diff(F, theta)
        Epsilon_F = theta*f/F

        ## show the plot
        theta =[]
        F_value =[]
        f_value = []
        Epsilon_F_value = []
        for i in np.arange(0,10,0.1):
            theta.append(i)
            F_value.append(float(F.subs('theta',i)))
            f_value.append(float(f.subs('theta', i)))
            Epsilon_F_value.append(float(Epsilon_F.subs('theta',i)))

        plt.plot(theta, F_value, "r-", linewidth=1, label='F')  # 画图
        plt.plot(theta, f_value, "y-", linewidth=1, label='f')  # 画图
        plt.plot(theta, Epsilon_F_value, "g-", linewidth=1, label='Epsilon_F')  # 画图
        plt.legend()
        plt.show()

        #calculate the value
        F_onevalue = float(F.subs('theta', self.cfg.THETA))
        f_onevalue = float(f.subs('theta', self.cfg.THETA))
        Epsilon_F_onevalue = self.cfg.THETA*f_onevalue/F_onevalue
        #print(F_onevalue)
        return F,f,Epsilon_F, F_onevalue, f_onevalue, Epsilon_F_onevalue

    def producer(self):
        '''
        self.p = the profit per platform consumer net of variable costs

        '''
        if (self.n >self.cfg.INFLECTION_POINT):
            p = self.cfg.C - self.cfg.D * self.cfg.INFLECTION_POINT
        else:
            p = self.cfg.C - self.cfg.D * self.n
        return p
    def producer_phi(self):

        phi = symbols("phi")
        H = 1/(1 + exp(-(phi-5))) #todo
        h = diff(H, phi)
        Epsilon_H = phi*h/H

        ## show the plot
        phi =[]
        H_value =[]
        h_value = []
        Epsilon_H_value = []
        for i in np.arange(0,10,0.1):
            phi.append(i)
            H_value.append(float(H.subs('phi',i)))
            h_value.append(float(h.subs('phi', i)))
            Epsilon_H_value.append(float(Epsilon_H.subs('phi',i)))

        plt.plot(phi, H_value, "r-", linewidth=1, label='H')  # 画图
        plt.plot(phi, h_value, "y-", linewidth=1, label='h')  # 画图
        plt.plot(phi, Epsilon_H_value, "g-", linewidth=1, label='Epsilon_H')  # 画图
        plt.legend()
        plt.show()

        #calculate the value
        H_onevalue = float(H.subs('phi', self.cfg.THETA))
        h_onevalue = float(h.subs('phi', self.cfg.THETA))
        Epsilon_H_onevalue = self.cfg.PHI*h_onevalue/H_onevalue
        #print(F_onevalue)
        return H,h,Epsilon_H, H_onevalue, h_onevalue, Epsilon_H_onevalue

    def profit(self,u,p):
        '''
        self.V = denote the gross surplus created by n products for each platform consumer
        '''
        V = u + p * self.n
        return V
    def profit_Epsilon(self):

        n = symbols("n")
        u = self.cfg.A + self.cfg.B * n
        p = self.cfg.C - self.cfg.D *n
        V = u + p*n ## n<=10
        v = diff(V, n)
        print('p(n)= ', p)
        print('u(n)= ', u)
        print('V(n)= ', V)
        print('v(n)= ', v)

        Epsilon_V = n * v / V
        lambda_n = p /v

        ## show the plot
        n = []
        V_value = []
        v_value = []
        Epsilon_V_value = []
        lambda_n_value =[]
        for i in np.arange(0, 10, 0.1):
            n.append(i)
            V_value.append(float(V.subs('n', i)))
            v_value.append(float(v.subs('n', i)))
            Epsilon_V_value.append(float(Epsilon_V.subs('n', i)))
            lambda_n_value.append(float(lambda_n.subs('n',i)))

        plt.plot(n, V_value, "r-", linewidth=1, label='V')
        plt.plot(n, v_value, "y-", linewidth=1, label='v')
        plt.legend()
        plt.show()
        plt.plot(n, Epsilon_V_value, "g-", linewidth=1, label='Epsilon_V')
        plt.plot(n, lambda_n_value, "b-", linewidth=1, label='lambda_n')
        plt.legend()
        plt.show()

        # calculate the value
        V_onevalue = float(V.subs('n', self.cfg.THETA))
        v_onevalue = float(v.subs('n', self.cfg.THETA))
        Epsilon_V_onevalue = self.cfg.PHI * v_onevalue / V_onevalue
        # print(F_onevalue)
        return u,p,V,v, V_onevalue, v_onevalue, Epsilon_V_onevalue

    def platform_profit(self,F):
        n, theta= symbols("n theta")

        plat_profit = self.cfg.PU*F +n*self.cfg.PD
        plat_profit_dif_n = diff(plat_profit,n)
        plat_profit_dif_theta = diff(plat_profit,theta)

        print('plat_profit = ',plat_profit)
        print('dif_n =',plat_profit_dif_n)
        print('dif_theta=',plat_profit_dif_theta)
        theta = []
        n=[]

        plat_profit_dif_n_value =[]
        plat_profit_dif_theta_value =[]
        for i in np.arange(0, 10, 0.1):
            n.append(i)
            plat_profit_dif_n_value.append(float(plat_profit_dif_n.subs('n',i)))
            theta.append(i)
            plat_profit_dif_theta_value.append(float(plat_profit_dif_theta.subs('theta', i)))


        plt.plot(theta, plat_profit_dif_theta_value, "r-", linewidth=1, label='dif_n')
        plt.plot(n, plat_profit_dif_n_value, "y-", linewidth=1, label='dif_theta')
        plt.legend()
        plt.show()

        return plat_profit, plat_profit_dif_n,plat_profit_dif_theta



    def platform_profit_show_in3D(self):
        '''
        here have to give the function of F again, here is
        :return:
        '''

        print('!!!!!!!!!!!!!')
        fig = plt.figure()
        ax = Axes3D(fig)
        n = np.arange(0,10,1)
        theta =  np.arange(0,10,1)
        n, theta = np.meshgrid(n,theta)
        F = 1/(np.exp(5-theta)+1)   # give the function of F again
        plat_profit =np.array( self.cfg.PU * F + n * self.cfg.PD)
        print(plat_profit)
        ax.plot_surface(n, theta, plat_profit, rstride=1, cstride=1, cmap='rainbow')
        plt.show()




    def visualisieren(self,n,p):
        print('#######')
        plt.figure()
        plt.plot(n, p)
        plt.show()




