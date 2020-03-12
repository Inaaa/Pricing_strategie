
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
AA=2

for i in np.arange(0.2, 1, 0.2):

    u = model.consumer(i,AA)
    p = model.producer(i,AA)
    V = model.profit(u, p)
    name =['n','u','p','V','u_p_v_n.png',color[int(i*5)],str(i)]
    model.visulation(name,u,p,V)
plt.legend()
plt.xlabel(name[0])
#plt.savefig(path + name[4])
plt.show()


