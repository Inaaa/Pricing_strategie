from two_side import *
#from config import *
from config2 import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
matplotlib.use( 'tkagg' )




cfg = two_side_config()
n = np.arange(0, 16, 1)


    model = BasisParameter(n, cfg)
    u =  model.producer()

'''
n = np.linspace(0,16,17)
print(n)
u = np.array([])
p = np.array([])
v = np.array([])
for i in n:
    model = BasisParameter(i,cfg)
    u = np.append(u, model.producer())
    p = np.append(p, model.customer())
    v = np.append(v, model.profit(u,p))
'''
#model.visualisieren(n,u)

plt.figure()
plt.plot(n, u, color="r", linestyle="--", linewidth=1.0)
plt.plot(n, p, color="g", linestyle="--", linewidth=1.0)
plt.plot(n, v, color="b", linestyle="-", linewidth=1.0)
plt.show()

'''
cfg = base_model_config()

mm = Pricing('bosch', cfg)
mm.profit_model()

'''