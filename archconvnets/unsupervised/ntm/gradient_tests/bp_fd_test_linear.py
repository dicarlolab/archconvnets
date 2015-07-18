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

x_cur = np.random.normal(size=(n_in,1))

mem_prev = np.random.normal(size=(M, mem_length))

wrkey = np.random.normal(size=(C,mem_length,n_in)) * SCALE * .5
o_key = np.random.normal(size=(C, mem_length)) * SCALE

O_KEY = np.random.normal(size=(4,5,5*8))
o_sq = np.random.normal(size=(4, 5))

t = np.random.normal(size=(C, M))

def f(y):
	mem_prev[i_ind,j_ind] = y
	
	o_content = linear_2d_F(O_KEY, mem_prev.ravel())
	
	return ((o_content - t)**2).sum()


def g(y):
	mem_prev[i_ind,j_ind] = y
	
	o_content = linear_2d_F(O_KEY, mem_prev.ravel())
	
	##
	derr_do_content = sq_points_dinput(o_content - t)
	
	do_content_dmem_prev = linear_2d_F_dx_nsum(O_KEY).reshape((4,5,5,8))
	
	derr_dmem_prev = mult_partials_sum(derr_do_content, do_content_dmem_prev, o_content)
	
	##
	
	return derr_dmem_prev[i_ind,j_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = mem_prev
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
