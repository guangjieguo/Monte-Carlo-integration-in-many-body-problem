import math
import numpy as np
from numpy import random as nr
import scipy
from scipy import optimize
import scipy.special as s
import gc
from time import *
from scipy import linalg


nk_matrix=[[6.12579002, 4.57713132, 3.12510319, 2.06011514, 1.35002685],
 [4.57713132, 4.11061461, 3.26157703, 2.4228063,  1.74541877],
 [3.12510319, 3.26157703, 2.93437125, 2.42218398, 1.90302311],
 [2.06011514, 2.4228063,  2.42218398, 2.18794899, 1.85675026],
 [1.35002685, 1.74541877, 1.90302311, 1.85675026, 1.68502635]]

h_matrix=[[38.25400227, 25.1316805,  15.27305198,  9.07098831,  5.41358115],
 [25.1316805,  21.09865394, 15.72188032, 11.02252718,  7.52901725],
 [15.27305198, 15.72188032, 13.92810335, 11.30381723,  8.72148561],
 [ 9.07098831, 11.02252718, 11.30381723, 10.40125945,  8.93454733],
 [ 5.41358115,  7.52901725,  8.72148561,  8.93454733,  8.42448359]]

result = linalg.eig(h_matrix,nk_matrix,left=False,right=True)

print(result)

eiv = min(result[0])

real_eiv = np.real(eiv)

index = np.argwhere(result[0]==eiv)[0][0]

print(real_eiv,result[1][:,index])

