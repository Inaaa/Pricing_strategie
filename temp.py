
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



n = np.arange(0, 16, 1)
u = np.zeros(16)
p = np.zeros(16)
v = np.zeros(16)
for i in n:
    model = BasisParameter(i, cfg,path)
    u[i] = model.consumer()
    p[i] = model.producer()
    v[i] = model.profit(u[i], p[i])
print(n,'++++',u)
plt.figure()
plt.plot(n, u, color="r", linestyle="--", linewidth=1.0, label='u')
plt.plot(n, p, color="g", linestyle="--", linewidth=1.0, label='p')
plt.plot(n, v, color="b", linestyle="-", linewidth=1.0, label='v')
plt.legend()
plt.xlabel('n')
plt.savefig(path+'u_p_v.png')
plt.show()



n = 5
model = BasisParameter(n,cfg,path)
F,f, Epsilon_F,_,_,Epsilon_F_onevalue = model.consumer_theta()
H,h,Epsilon_H,_,_,Epsilon_H_onevalue = model.producer_phi()
u, p, V, v,lambda_n,Epsilon_V, V_onevalue,v_onevalue, Epsilon_V_onevalue,lambda_n_onevalue = model.profit_Epsilon()
plat_profit,_,_ = model.platform_profit(F)
model.platform_profit_show_in3D()
model.platform_lambda(Epsilon_H,Epsilon_F,Epsilon_V)
model.producer_profit_to_constumer_profit(Epsilon_H,Epsilon_F,lambda_n,Epsilon_V)

model.producer_profit_to_constumer_profit(Epsilon_H_onevalue,Epsilon_F_onevalue,lambda_n_onevalue,Epsilon_V_onevalue)

#print(F)


## plot producer theta

