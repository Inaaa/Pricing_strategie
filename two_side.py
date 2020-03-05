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

    def __init__(self,n, cfg2,path):
        '''
        path--save the image
        '''
        self.n = n
        self.cfg = cfg2
        self.path = path
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
        print(type(F))

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

        plt.plot(theta, F_value, "r-", linewidth=1, label='F='+str(F))  # 画图
        plt.plot(theta, f_value, "y-", linewidth=1, label='f='+str(f))  # 画图
        plt.plot(theta, Epsilon_F_value, "g-", linewidth=1, label='Epsilon_F='+str(Epsilon_F))  # 画图
        plt.legend()
        plt.xlabel('theta')
        plt.savefig(self.path+'consumer_theta.png')
        plt.show()

        #calculate the value
        F_onevalue = float(F.subs('theta', self.cfg.THETA))
        f_onevalue = float(f.subs('theta', self.cfg.THETA))
        Epsilon_F_onevalue = self.cfg.THETA*f_onevalue/F_onevalue
        #print(F_onevalue)
        print('coustomer_theta F=',F)
        print('coustomer_theta f=', f)
        print('coustomer_theta Epsilon_F=', Epsilon_F)
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

        plt.plot(phi, H_value, "r-", linewidth=1, label='H='+str(H))  # 画图
        plt.plot(phi, h_value, "y-", linewidth=1, label='h='+str(h))  # 画图
        plt.plot(phi, Epsilon_H_value, "g-", linewidth=1, label='Epsilon_H='+str(Epsilon_H))  # 画图
        plt.legend()
        plt.xlabel('phi')
        plt.savefig(self.path+'producer_phi.png')
        plt.show()

        #calculate the value
        H_onevalue = float(H.subs('phi', self.cfg.PHI))
        h_onevalue = float(h.subs('phi', self.cfg.PHI))
        Epsilon_H_onevalue = self.cfg.PHI*h_onevalue/H_onevalue
        #print(F_onevalue)
        print('producer_phi H=', H)
        print('producer_phi h=', h)
        print('producer_phi Epsilon_H=', Epsilon_H)
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
        print('profit_Epsilon Epsilon_V=',Epsilon_V)
        print('profit_Epsilon lambda_n=', lambda_n)

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

        plt.plot(n, V_value, "r-", linewidth=1, label='V ='+str(V))
        plt.plot(n, v_value, "y-", linewidth=1, label='v='+str(v))
        plt.legend()
        plt.xlabel('n')
        plt.savefig(self.path+'profit_V.png')
        plt.show()
        plt.plot(n, Epsilon_V_value, "g-", linewidth=1, label='Epsilon_V='+str(Epsilon_V))
        plt.plot(n, lambda_n_value, "b-", linewidth=1, label='lambda_n='+str(lambda_n))
        plt.xlabel('n')
        plt.legend()
        plt.savefig(self.path + 'Epsilon_V+lambda_n.png')
        plt.show()

        # calculate the value
        V_onevalue = float(V.subs('n', self.cfg.N))
        v_onevalue = float(v.subs('n', self.cfg.N))
        Epsilon_V_onevalue = self.cfg.PHI * v_onevalue / V_onevalue
        lambda_n_onevalue = float(lambda_n.subs('n',self.cfg.N))
        # print(F_onevalue)
        return u,p,V,v, lambda_n,Epsilon_V, V_onevalue, v_onevalue, Epsilon_V_onevalue,lambda_n_onevalue

    def platform_profit(self,F):
        n, theta= symbols("n theta")

        plat_profit = self.cfg.PU*F +n*self.cfg.PD
        plat_profit_dif_n = diff(plat_profit,n)
        plat_profit_dif_theta = diff(plat_profit,theta)

        print('plat_profit = ',plat_profit)
        print('plat_profit_dif_n =',plat_profit_dif_n)
        print('plat_profit_dif_theta=',plat_profit_dif_theta)
        theta = []
        n=[]

        plat_profit_dif_n_value =[]
        plat_profit_dif_theta_value =[]

        for i in np.arange(0, 10, 0.1):
            n.append(i)
            plat_profit_dif_n_value.append(float(plat_profit_dif_n.subs('n',i)))
            theta.append(i)
            plat_profit_dif_theta_value.append(float(plat_profit_dif_theta.subs('theta', i)))


        plt.plot(theta, plat_profit_dif_theta_value, "r-", linewidth=1, label='dif_n/theta ='+str(plat_profit_dif_theta))
        plt.plot(n, plat_profit_dif_n_value, "y-", linewidth=1, label='dif_theta/n='+str(plat_profit_dif_n))
        plt.xlabel('n or theta')
        plt.legend()
        plt.savefig(self.path + 'platform_profit.png')
        plt.show()


        #f = lambdify((n,theta),plat_profit)wuhan

        return plat_profit, plat_profit_dif_n,plat_profit_dif_theta

    def producer_profit_to_constumer_profit(self,Epsilon_H,Epsilon_F,lambda_n,Epsilon_V):

        producer_profit_to_constumer_profit = Epsilon_V*(1+Epsilon_F)*(1-(1-lambda_n)*(1+Epsilon_H))/ \
                                              (1+Epsilon_H)*(1-lambda_n*Epsilon_V*(1+Epsilon_F))


        print(type(producer_profit_to_constumer_profit))

        print(producer_profit_to_constumer_profit)
        return(producer_profit_to_constumer_profit)

    def calculate(self,producer_profit_to_constomer_profit):

        n=10
        phi=5
        theta=5





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
        print(type(F))
        plat_profit =np.array( self.cfg.PU * F + n * self.cfg.PD)

        #print(plat_profit)
        ax.plot_surface(n, theta, plat_profit, rstride=1, cstride=1, cmap='rainbow')
        plt.xlabel('n')
        plt.ylabel('theta')
        plt.savefig(self.path + 'platform_profit3D.png')
        plt.show()

    def platform_lambda(self,Epsilon_H,Epsilon_F,Eplison_V):
        n, theta,phi = symbols("n theta phi")
        lambda_max = 1/Eplison_V * (1 + Epsilon_F)
        lambda_min = Epsilon_H / (1 + Epsilon_H)

        lambda_max_value = []
        lambda_min_value = []
        print('lambda_min=', lambda_min)
        print('lambda_max =',lambda_max)
        '''
        theta = []
        phi=[]
        lambda_max_value =[]
        lambda_min_value = []
        plat_profit_dif_theta_value =[]
        for i in np.arange(0, 10, 0.1):
            phi.append(i)
            lambda_min_value.append(float(lambda_min.subs('phi',i)))
            theta.append(i)
            lambda_max_value.append(float(lambda_max.subs('theta', i)))


        plt.plot(phi,lambda_min_value, "r-", linewidth=1, label='lambda_min/phi='+str(lambda_min))
        plt.plot(theta,lambda_max_value, "y-", linewidth=1, label='lambda_max/theta='+str(lambda_max))
        plt.xlabel('phi or theta')
        plt.legend()
        plt.savefig(self.path + 'platform_lambda.png')
        plt.show()
        '''




    def visualisieren(self,n,p):
        print('#######')
        plt.figure()
        plt.plot(n, p)
        plt.show()




