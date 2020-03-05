
from two_side import *
#from config import *
from config2 import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
matplotlib.use( 'tkagg' )



# plot the u(n),p(n),v(n)
cfg = two_side_config()
path ='/home/chli/Documents/hiwi/Pricing_strategie/result/'

model = BasisParameter(cfg,path)

u = model.consumer()
p = model.producer()
V = model.profit(u, p)
name =['n','u','p','V','u_p_v_n.png']
model.visulation(name,u,p,V)


F,f, Epsilon_F,_,_,Epsilon_F_onevalue = model.consumer_theta()

name =['theta','F','f','Epsilon_F','consumer_theta.png']
model.visulation(name,F,f,Epsilon_F)

H,h,Epsilon_H,_,_,Epsilon_H_onevalue = model.producer_phi()
name =['phi','H','h','Epsilon_H','producer_phi.png']
model.visulation(name,H,h,Epsilon_H)

v,lambda_n,Epsilon_V, V_onevalue,v_onevalue, Epsilon_V_onevalue,lambda_n_onevalue = model.profit_Epsilon(p,V)
name = ['n','V','v','profit_V.png']
model.visulation2(name,V,v)
name =['n','Epsilon_V','lambda_n','Epsilon_V+lambda_n.png']
model.visulation2(name,Epsilon_V,lambda_n)

plat_profit,plat_profit_dif_n,plat_profit_dif_theta = model.platform_profit(F)
model.platform_profit_show_in3D()

lambda_min,lambda_max=model.platform_lambda(Epsilon_H,Epsilon_F,Epsilon_V)
producer_profit_to_constumer_profit= model.producer_profit_to_constumer_profit(Epsilon_H,Epsilon_F,lambda_n,Epsilon_V)

model.producer_profit_to_constumer_profit(Epsilon_H_onevalue,Epsilon_F_onevalue,lambda_n_onevalue,Epsilon_V_onevalue)


print('u=',u)
print('p=',p)
print('V=',V)
print('v=',v)
print('coustomer_theta F=', F)
print('coustomer_theta f=', f)
print('coustomer_theta Epsilon_F=', Epsilon_F)

print('producer_phi H=', H)
print('producer_phi h=', h)
print('producer_phi Epsilon_H=', Epsilon_H)

print('profit_Epsilon Epsilon_V=', Epsilon_V)
print('profit_Epsilon lambda_n=', lambda_n)

print('plat_profit = ', plat_profit)
print('plat_profit_dif_n =', plat_profit_dif_n)
print('plat_profit_dif_theta=', plat_profit_dif_theta)
print('producer_profit_to_constumer_profit',producer_profit_to_constumer_profit)
print('lambda_min=', lambda_min)
print('lambda_max =', lambda_max)

## plot producer theta

