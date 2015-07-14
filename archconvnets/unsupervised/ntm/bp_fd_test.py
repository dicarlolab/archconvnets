import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

C = 4
M = 5
n_in = 3
n_shifts = 3
mem_length = 8

SCALE = .6

o_sq = np.random.normal(size=(C, M))
wrkey = np.random.normal(size=(C,mem_length,n_in)) * SCALE * .5
mem_previ = np.random.normal(size=(M, mem_length))
z = np.random.normal(size=(C, mem_length))
t = np.random.normal(size=(C, M))

def f(y):
	#z[i_ind,j_ind] = y
	mem_previ[i_ind,j_ind] = y
	
	o = linear_F(z,mem_previ.T)
	
	return ((o - t)**2).sum()


def g(y):
	#z[i_ind,j_ind] = y
	mem_previ[i_ind,j_ind] = y
	
	o = linear_F(z,mem_previ.T)
	
	derr_do = sq_points_dinput(o - t)
	
	do_dkeys = linear_F_dF_nsum_g(z, mem_previ.T)
	do_dmem = linear_F_dx_nsum_g(z, mem_previ.T)
	
	dz = mult_partials_sum(derr_do, do_dkeys, o)
	dmem = mult_partials_sum(derr_do, do_dmem, o).T
	
	return dmem[i_ind,j_ind]
	#return dz[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = mem_previ
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
