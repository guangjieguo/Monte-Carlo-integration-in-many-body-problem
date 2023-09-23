import math
import numpy as np
from numpy import random as nr
import scipy
from scipy import optimize
import scipy.special as s
import gc
from time import *
import matplotlib.pyplot as plt


# 参数模块
n=4
g=2
beta=[math.sqrt(2/n)-0.1,math.sqrt(2/n),math.sqrt(2/n)+0.1]
c = np.array([0.0766383,0.530031,0.844508])
nk = np.array([[4.13133514,3.27469665,2.43107694],
 [3.27469665,2.94807196,2.4316953 ],
 [2.43107694,2.4316953 ,2.19708154]])

"""
"""
num_beta=len(beta)

P=n*(n-1)/2

def f(x,beta_i):
    func=(g/beta_i)*(s.gamma(-x/2)/s.gamma((1-x)/2))+2**(3/2)
    return func

mu=[]
a=[]
for i in range(num_beta):
    root=optimize.fsolve(f,0.5,args=(beta[i]))[0]
    mu.append(root)
    a.append(-(root+1/2))

def psiR(x):
    wavefunc=np.e**(-(n/2)*(np.sum(x, axis=1)/n)**2)
    return wavefunc

def psicp(beta_i,a_i,x):
    wavefunc=1
    for i in range(n):
        for j in range(n):
            if i<j:
                wavefunc=wavefunc*2**(-1/4-a_i/2)*np.e**(-(1/4)*beta_i**2*(x[:,j]-x[:,i])**2
                            )*s.hyperu(a_i/2+1/4, 1/2, (1/2)*beta_i**2*(x[:,j]-x[:,i])**2, out=None)
    return wavefunc

def psi(beta_i,a_i,x):
    wavefunc=psiR(x)*psicp(beta_i,a_i,x)
    return wavefunc

def total_wf(nk,c,x):
    normal_c = 0
    for i in range(num_beta):
        for j in range(num_beta):
            normal_c += nk[i,j] * c[i] * c[j]
    wf=0
    for i in range(num_beta):
        wf += c[i]*psi(beta[i],a[i],x)
    return (1/math.sqrt(normal_c))*wf

def integ_func(nk,c,x):
    return total_wf(nk,c,x)*total_wf(nk,c,x)

# 这个有问题
def assis_func(sigma,x):
        return np.e**((np.sum(x*x, axis=1))/(sigma**2))

# 得到随机数
num=10000000
points = nr.uniform(0,1, (num, n-1))

sigma = 5/max(beta)
integ_domin = 10/min(beta)
lim_bottom = sigma*math.sqrt(math.pi)*(1-s.erf(integ_domin/sigma))/2
lim_top = sigma*math.sqrt(math.pi)*(1+s.erf(integ_domin/sigma))/2

r_points_low = (lim_top-lim_bottom) * points + lim_bottom
       
r_points_low = sigma*s.erfinv((2/(sigma*math.sqrt(math.pi))) * r_points_low - 1)

assis_func_values = assis_func(sigma,r_points_low)

def cal_onebodydensity(nk,c,x1):

    x1_array = np.full(num,x1)
    x1_array = x1_array[:,np.newaxis]
    r_points = np.append(r_points_low,x1_array,axis=1)

    average_trans_func1 = np.sum(integ_func(nk,c,r_points)*assis_func_values)/num
    
    density = ((lim_top-lim_bottom)**(n-1)) * average_trans_func1

    return density

data_density = []
data_xlabel = []
for i in range(1,71):
    x1 = i*0.05
    data_density.append(cal_onebodydensity(nk,c,x1))
    data_xlabel.append(x1)

print(data_density)

"""
#drawing plots
plt.plot(data_xlabel,data_density,linewidth=1)

data2 = [[0.04562043795620435, 0.39274447949526814],
  [0.12226277372262773, 0.39085173501577286],
  [0.19616788321167888, 0.38832807570977923],
  [0.2728102189781021, 0.3839116719242902],
  [0.3494525547445255, 0.37823343848580443],
  [0.4233576642335767, 0.3712933753943218],
  [0.4972627737226277, 0.36309148264984226],
  [0.5766423357664234, 0.3529968454258675],
  [0.6505474452554745, 0.3416403785488959],
  [0.7271897810218979, 0.328391167192429],
  [0.8038321167883213, 0.3138801261829653],
  [0.8804744525547445, 0.2974763406940063],
  [0.957116788321168, 0.27917981072555204],
  [1.0364963503649636, 0.2589905362776025],
  [1.1131386861313868, 0.23817034700315454],
  [1.1925182481751826, 0.21608832807570977],
  [1.2718978102189782, 0.19337539432176654],
  [1.351277372262774, 0.1706624605678233],
  [1.4333941605839418, 0.14921135646687694],
  [1.5127737226277373, 0.12776025236593053],
  [1.5948905109489053, 0.10757097791798098],
  [1.677007299270073, 0.08927444794952677],
  [1.759124087591241, 0.07287066246056773],
  [1.8439781021897816, 0.05772870662460561],
  [1.9288321167883213, 0.045741324921135584],
  [2.016423357664234, 0.03564668769716084],
  [2.101277372262774, 0.02681388012618291],
  [2.1916058394160585, 0.01987381703470026],
  [2.279197080291971, 0.014195583596214423],
  [2.3722627737226283, 0.010410094637223977],
  [2.468065693430657, 0.007255520504731772],
  [2.5583941605839424, 0.005362776025236493],
  [2.6596715328467155, 0.0028391167192427513],
  [2.7609489051094895, 0.001577287066245936],
  [2.8649635036496353, 0.001577287066245936],
  [2.974452554744526, 0.001577287066245936]]
x_values = []
y_values = []
for i in range(len(data2)):
    x_values.append(data2[i][0])
    y_values.append(data2[i][1])

plt.scatter(x_values,y_values,c='red',s=5)

plt.show()

"""








