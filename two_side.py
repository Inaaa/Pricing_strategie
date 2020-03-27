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

    def __init__(self, cfg2,path):
        '''
        path--save the image
        '''

        self.cfg = cfg2
        self.path = path
        #print(self.cfg)
    def consumer(self,*args):
        '''
        self.u =  surplus obtained  from the n products;  linear function
        define the inflection point n=10

        '''
        n = symbols('n')
        if len(args) == 0:
            u = self.cfg.A + self.cfg.B * n
        elif len(args)== 2:

            #u = (1-self.cfg.BETA)*self.cfg.AA*pow(n,self.cfg.BETA)
            u = (1 - args[0]) * args[1] * pow(n, args[0])
            # example 1
        else:
            ## 0< alpha <theta<1
            #u = (1- alpha)*pow((alpha *theta /c ),(alpha/(1-alpha)))*pow(n, alpha*(1-theta)/(theta*(1-alpha)))
            par1=(1- args[0])*pow((args[0]*args[1]/args[2]),(args[0]/(1-args[0])))
            par1=round(par1, 5 - len(str(int(par1)))) if len(str(par1)) > 5 + 1 else par1
            u = par1*pow(n,args[0]*(1-args[1])/(args[1]*(1-args[0])))




        print(args)
        print(len(args))
        #u = (1-beta)*a*pow(n,beta)
        #self.epsilon = theta *self.f/self.F
        return u

    def consumer_theta(self):

        theta = symbols("theta")
        F = 1/(1 + exp(-(theta-5)))
        f = diff(F, theta)
        Epsilon_F = theta*f/F

        #calculate the value
        F_onevalue = float(F.subs('theta', self.cfg.THETA))
        f_onevalue = float(f.subs('theta', self.cfg.THETA))
        Epsilon_F_onevalue = self.cfg.THETA*f_onevalue/F_onevalue
        #print(F_onevalue)

        return F,f,Epsilon_F, F_onevalue, f_onevalue, Epsilon_F_onevalue

    def producer(self,*args):
        '''
        self.p = the profit per platform consumer net of variable costs

        '''
        n = symbols('n')
        if len(args) == 0:
            p = self.cfg.C - self.cfg.D * n
        elif len(args) ==2:
            #p = self.cfg.BETA*self.cfg.AA*pow(n,self.cfg.BETA-1)
            p = args[0] * args[1] * pow(n, args[0] - 1)
        else:
            #p = (1 - theta)*alpha*pow((alpha*theta/c),(alpha/(1-alpha)))* pow(n,-1*(theta-alpha)/(theta *(1-alpha)))
            par1=(1-args[1])*args[0]*pow((args[0]*args[1]/args[2]),(args[0]/(1-args[0])))
            par1=round(par1, 5 - len(str(int(par1)))) if len(str(par1)) > 5 + 1 else par1
            p = par1*pow(n, -1*(args[1]-args[0])/(args[1]*(1-args[0])))

        return p

    def producer_phi(self):

        phi = symbols("phi")
        H = 1/(1 + exp(-(phi-5))) #todo
        h = diff(H, phi)
        Epsilon_H = phi*h/H

        #calculate the value
        H_onevalue = float(H.subs('phi', self.cfg.PHI))
        h_onevalue = float(h.subs('phi', self.cfg.PHI))
        Epsilon_H_onevalue = self.cfg.PHI*h_onevalue/H_onevalue
        #print(F_onevalue)

        return H,h,Epsilon_H, H_onevalue, h_onevalue, Epsilon_H_onevalue

    def profit(self,u,p):
        '''
        self.V = denote the gross surplus created by n products for each platform consumer
        '''
        n = symbols('n')
        V = u + p *n
        return V
    def profit_Epsilon(self,p,V):

        n = symbols("n")
        v = diff(V, n)
        Epsilon_V = n * v / V
        lambda_n = p /v

        # calculate the value
        V_onevalue = float(V.subs('n', self.cfg.N))
        v_onevalue = float(v.subs('n', self.cfg.N))
        Epsilon_V_onevalue = self.cfg.PHI * v_onevalue / V_onevalue
        lambda_n_onevalue = float(lambda_n.subs('n',self.cfg.N))
        # print(F_onevalue)
        return v, lambda_n,Epsilon_V, V_onevalue, v_onevalue, Epsilon_V_onevalue,lambda_n_onevalue

    def platform_profit(self,F,*args):
        n, theta= symbols("n theta")

        plat_profit = self.cfg.PU*F +n*self.cfg.PD
        plat_profit_dif_n = diff(plat_profit,n)
        plat_profit_dif_theta = diff(plat_profit,theta)

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

        return(producer_profit_to_constumer_profit)

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
        return lambda_min,lambda_max


    def visulation(self,args,x1,x2,x3):

        x1_value = []
        x2_value = []
        x3_value = []
        x_value= []
        for i in np.arange(0, 10, 0.1):
            x_value.append(i)
            x1_value.append(float(x1.subs(args[0], i)))
            x2_value.append(float(x2.subs(args[0], i)))
            x3_value.append(float(x3.subs(args[0], i)))

        if len(args) == 5:
            plt.figure()
            plt.plot(x_value, x1_value, "r-", linewidth=1, label=args[1]+'='+str(x1))  # 画图
            plt.plot(x_value, x2_value, "y-", linewidth=1, label=args[2]+'='+str(x2))  # 画图
            plt.plot(x_value, x3_value, "g-", linewidth=1, label=args[3]+'='+str(x3))  # 画图
            plt.legend()
            plt.xlabel(args[0])
            plt.savefig(self.path + args[4])
            plt.show()

        else:
            plt.plot(x_value, x1_value, args[5] + "--", linewidth=1,
                     label=args[1] + '=' + str(x1) + '  beta =' + args[6][0:3])  # 画图
            plt.plot(x_value, x2_value, args[5] + "-", linewidth=1, label=args[2] + '=' + str(x2))  # 画图
            plt.plot(x_value, x3_value, args[5] + ".", linewidth=1, label=args[3] + '=' + str(x3))  # 画图



    def visulation2(self,args,x1,x2):



        x1_value = []
        x2_value = []
        x_value= []
        for i in np.arange(0, 10, 0.1):
            x_value.append(i)
            x1_value.append(float(x1.subs(args[0], i)))
            x2_value.append(float(x2.subs(args[0], i)))

        if len(args) == 4:
            plt.figure()
            plt.plot(x_value, x1_value, "r-", linewidth=1, label=args[1]+'='+str(x1))  # 画图
            plt.plot(x_value, x2_value, "y-", linewidth=1, label=args[2]+'='+str(x2))  # 画图
            plt.legend()
            plt.xlabel(args[0])
            plt.savefig(self.path + args[3])
            plt.show()
        else:
            plt.plot(x_value, x1_value, args[4] + "--", linewidth=1,
                     label=args[1] + '=' + str(x1)[0:4] + '  beta =' + args[5][0:3])  # 画图
            plt.plot(x_value, x2_value, args[4] + "-", linewidth=1, label=args[2] + '=' + str(x2)[0:4])





