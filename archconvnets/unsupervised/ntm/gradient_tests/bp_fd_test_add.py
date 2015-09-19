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

mem_prev = np.random.normal(size=(M, mem_length))

owf = np.random.normal(size=(C,M)) * SCALE * .5
add_out = np.random.normal(size=(C, mem_length)) * SCALE

t = np.random.normal(size=(M, mem_length))

def f(y):
	#owf[i_ind,j_ind] = y
	add_out[i_ind,j_ind] = y
	
	o = add_mem(owf, add_out)
	
	return ((o - t)**2).sum()


def g(y):
	#owf[i_ind,j_ind] = y
	add_out[i_ind,j_ind] = y
	
	o = add_mem(owf, add_out)
	
	##
	derr_do = sq_points_dinput(o - t)
	
	do_dowf = add_mem_dgw(add_out)
	do_dadd_out = add_mem_dadd_out(owf)
	
	derr_dowf = mult_partials_sum(derr_do, do_dowf, o)
	derr_dadd_out = mult_partials_sum(derr_do, do_dadd_out, o)
	
	##
	
	return derr_dadd_out[i_ind,j_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	#ref = owf
	ref = add_out
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
