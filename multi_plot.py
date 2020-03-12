
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
path ='/home/chli/Documents/hiwi/Pricing_strategie/result3/'

model = BasisParameter(cfg,path)

color =("","r","b","g","y","c")


## visualation u_p_v_n.png
for i in np.arange(0.2, 1, 0.2):

    u = model.consumer(i,2)
    p = model.producer(i,2)
    V = model.profit(u, p)
    name =['n','u','p','V','u_p_v_n.png',color[int(i*5)],str(i)]
    model.visulation(name,u,p,V)
plt.legend()
plt.xlabel(name[0])
plt.savefig(path + name[4])
plt.show()





F,f, Epsilon_F,_,_,Epsilon_F_onevalue = model.consumer_theta()

name =['theta','F','f','Epsilon_F','consumer_theta.png']
model.visulation(name,F,f,Epsilon_F)

H,h,Epsilon_H,_,_,Epsilon_H_onevalue = model.producer_phi()
name =['phi','H','h','Epsilon_H','producer_phi.png']
model.visulation(name,H,h,Epsilon_H)



# visualation profit_v.png
for i in np.arange(0.2, 1, 0.2):

    u = model.consumer(i,2)
    p = model.producer(i,2)
    V = model.profit(u, p)
    v, lambda_n, Epsilon_V, V_onevalue, v_onevalue, Epsilon_V_onevalue, lambda_n_onevalue = model.profit_Epsilon(p, V)
    name = ['n', 'V', 'v', 'profit_V.png',color[int(i*5)],str(i)]
    model.visulation2(name, V, v)
plt.legend()
plt.xlabel(name[0])
plt.savefig(path + name[3])
plt.show()



# visualation Epsilon_V+lambda_n.png
for i in np.arange(0.2, 1, 0.2):

    u = model.consumer(i,2)
    p = model.producer(i,2)
    V = model.profit(u, p)
    v, lambda_n, Epsilon_V, V_onevalue, v_onevalue, Epsilon_V_onevalue, lambda_n_onevalue = model.profit_Epsilon(p, V)
    name = ['n','Epsilon_V','lambda_n','Epsilon_V+lambda_n.png',color[int(i*5)],str(i)]
    model.visulation2(name, Epsilon_V,lambda_n)
plt.legend()
plt.xlabel(name[0])
plt.savefig(path + name[3])
plt.show()




## plot producer theta
