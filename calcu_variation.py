import math
import numpy as np
from numpy import random as nr
import scipy
from scipy import optimize
import scipy.special as s
import gc
from time import *


# 参数模块
n=3
g=2
beta=[]
for i in range(20):
    beta.append(math.sqrt(2/n)+0.01*2+0.001*i)

num=50000000

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

def T1(beta_j,a_j,x):
    term=0
    psicpx=psicp(beta_j,a_j,x)
    for i in range(n):
        for j in range(n):
            if i<j:
                term=term + (x[:,i]-x[:,j])**2 * psicpx
    term=term*((2-n*beta_j**4)/(4*n))
    return term

def psicp_klpq(k,l,p,q,beta_j,a_j,x):
    wf=1
    for i in range(n):
        for j in range(n):
            if i<j and (not (i==k and j==l)) and (not (i==p and j==q)):
                wf = wf * (2**(-1/4-a_j/2) * np.e**(-(1/4) * beta_j**2 * (x[:,j]-x[:,i])**2) * s.hyperu(a_j/2+1/4, 1/2, (1/2)*beta_j**2*(x[:,j]-x[:,i])**2, out=None))
    return wf

def T2(beta_j,a_j,x):
    term=0
    for k in range(n):
        for l in range(n):
            if k<l:
                for p in range(n):
                    for q in range(n):
                        if p<q and (not (k==p and l==q)):
                            if l==p or k==q:
                                term += psicp_klpq(k,l,p,q,beta_j,a_j,x) * (-0.5*2**(-0.25-0.5*a_j)*np.e**(-0.25*(beta_j*(x[:,k]-x[:,l]))**2)*beta_j*(x[:,k]-x[:,l])*s.hyperu(
                                    a_j/2+1/4, 1/2, (1/2)*beta_j**2*(x[:,k]-x[:,l])**2, out=None) + 2**(-0.25-0.5*a_j)*np.e**(-0.25*(beta_j*(x[:,k]-x[:,l]))**2)*(
                                        -(0.25+0.5*a_j)*beta_j*(x[:,k]-x[:,l])*s.hyperu(a_j/2+5/4, 3/2, (1/2)*beta_j**2*(x[:,k]-x[:,l])**2, out=None))) * (
                                            -0.5*2**(-0.25-0.5*a_j)*np.e**(-0.25*(beta_j*(x[:,p]-x[:,q]))**2)*beta_j*(x[:,p]-x[:,q])*s.hyperu(a_j/2+1/4, 1/2, (1/2)*beta_j**2*(x[:,p]-x[:,q])**2, out=None) + 2**(
                                         -0.25-0.5*a_j)*np.e**(-0.25*(beta_j*(x[:,p]-x[:,q]))**2)*(-(0.25+0.5*a_j)*beta_j*(x[:,p]-x[:,q])*s.hyperu(a_j/2+5/4, 3/2, (1/2)*beta_j**2*(x[:,p]-x[:,q])**2, out=None)))
                            if k==p or l==q:
                                term -= psicp_klpq(k,l,p,q,beta_j,a_j,x) * (-0.5*2**(-0.25-0.5*a_j)*np.e**(-0.25*(beta_j*(x[:,k]-x[:,l]))**2)*beta_j*(x[:,k]-x[:,l])*s.hyperu(
                                    a_j/2+1/4, 1/2, (1/2)*beta_j**2*(x[:,k]-x[:,l])**2, out=None) + 2**(-0.25-0.5*a_j)*np.e**(-0.25*(beta_j*(x[:,k]-x[:,l]))**2)*(
                                        -(0.25+0.5*a_j)*beta_j*(x[:,k]-x[:,l])*s.hyperu(a_j/2+5/4, 3/2, (1/2)*beta_j**2*(x[:,k]-x[:,l])**2, out=None))) * (
                                            -0.5*2**(-0.25-0.5*a_j)*np.e**(-0.25*(beta_j*(x[:,p]-x[:,q]))**2)*beta_j*(x[:,p]-x[:,q])*s.hyperu(a_j/2+1/4, 1/2, (1/2)*beta_j**2*(x[:,p]-x[:,q])**2, out=None) + 2**(
                                         -0.25-0.5*a_j)*np.e**(-0.25*(beta_j*(x[:,p]-x[:,q]))**2)*(-(0.25+0.5*a_j)*beta_j*(x[:,p]-x[:,q])*s.hyperu(a_j/2+5/4, 3/2, (1/2)*beta_j**2*(x[:,p]-x[:,q])**2, out=None)))

    term = (beta_j**(2)/2) * term
    return term

def func1(i,j,x):
    return psi(beta[i],a[i],x)*psi(beta[j],a[j],x)

def func2(i,j,x):
    return psi(beta[i],a[i],x)*(T1(beta[j],a[j],x)+T2(beta[j],a[j],x))*psiR(x)

def assis_func(sigma,x):
        return np.e**((np.sum(x*x, axis=1))/(sigma**2))

# 得到随机数
points = nr.uniform(0,1, (num, n))

# 计算能量
def cal_energy(i):
    sigma = 5/beta[i]
    integ_domin = 10/beta[i]
    lim_bottom = sigma*math.sqrt(math.pi)*(1-s.erf(integ_domin/sigma))/2
    lim_top = sigma*math.sqrt(math.pi)*(1+s.erf(integ_domin/sigma))/2

    r_points = (lim_top-lim_bottom) * points + lim_bottom
    
    r_points = sigma*s.erfinv((2/(sigma*math.sqrt(math.pi))) * r_points - 1)

    average_trans_func1 = np.sum(func1(i,i,r_points)*assis_func(sigma,r_points))/num
    
    nk_ele = ((lim_top-lim_bottom)**n) * average_trans_func1

    trans_func2_values = func2(i,i,r_points)*assis_func(sigma,r_points)
    sum_f2=0
    total_num = num
    for j in range(num):
        if not np.isnan(trans_func2_values[j]):
            sum_f2 = sum_f2 + trans_func2_values[j]
        else:
            total_num = total_num - 1
    average_trans_func2 = sum_f2/total_num
    energy = (1/2 + beta[i]**2*P*(mu[i]+1/2))*nk_ele + ((lim_top-lim_bottom)**n)*average_trans_func2

    return energy/nk_ele

energy_matrix = []

for i in range(num_beta):
    energy_matrix.append(cal_energy(i))

print(energy_matrix)
