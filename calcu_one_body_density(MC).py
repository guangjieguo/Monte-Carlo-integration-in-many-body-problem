import math
import numpy as np
from numpy import random as nr
import scipy
from scipy import optimize
import scipy.special as s
import gc
from time import *
import matplotlib.pyplot as plt


# parameters setting
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

# define the auxiliary function
def assis_func(sigma,x):
        return np.e**((np.sum(x*x, axis=1))/(sigma**2))

# initiate random points for Monte Carlo integration
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








