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

wrshift = np.random.normal(size=(C,n_shifts,n_in)) * SCALE * .5

o_sq = np.random.normal(size=(4, 5))

t = np.random.normal(size=(C, M))

def f(y):
	wrshift[i_ind,j_ind,k_ind] = y
	#o_sq[i_ind,j_ind] = y
	
	o_shift = linear_2d_F(wrshift, x_cur)
	o = shift_w(o_shift, o_sq)
	
	return ((o - t)**2).sum()


def g(y):
	wrshift[i_ind,j_ind,k_ind] = y
	#o_sq[i_ind,j_ind] = y
	
	o_shift = linear_2d_F(wrshift, x_cur)
	o = shift_w(o_shift, o_sq)
	
	##
	derr_do = sq_points_dinput(o - t)
	
	do_do_shift = shift_w_dshift_out_nsum(o_sq)
	do_do_sq = shift_w_dw_interp_nsum(o_shift)
	
	do_shift_dwrshift = linear_2d_F_dF_nsum(wrshift, x_cur)
	
	do_dwrshift = mult_partials(do_do_shift, do_shift_dwrshift, o_shift)
	
	derr_dwrshift = mult_partials_sum(derr_do, do_dwrshift, o)
	derr_do_sq = mult_partials_sum(derr_do, do_do_sq, o)
	
	return derr_dwrshift[i_ind,j_ind,k_ind]
	#return derr_do_sq[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	#ref = o_sq
	ref = wrshift
	#i_ind = np.random.randint(ref.shape[0])
	#j_ind = np.random.randint(ref.shape[1])
	#y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	k_ind = np.random.randint(ref.shape[2])
	y = -1e0*ref[i_ind,j_ind,k_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
