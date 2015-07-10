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
n_in = 3
n_shifts = 3
M = 5

o_sq = np.random.normal(size=(C, M))
shift_out = np.random.normal(size=(C, n_shifts))

t = np.random.normal(size=(C, M))

def f(y):
	#o_sq[i_ind,j_ind] = y
	shift_out[i_ind,j_ind] = y
	
	o = shift_w(shift_out, o_sq)
	
	return ((o - t)**2).sum()


def g(y):
	#o_sq[i_ind,j_ind] = y
	shift_out[i_ind,j_ind] = y
	
	o = shift_w(shift_out, o_sq)
	
	derr_do = sq_points_dinput(o - t)
	
	do_do_sq = shift_w_dw_interp_nsum(shift_out)
	do_dshift_out = shift_w_dshift_out_nsum(o_sq)
	
	derr_do_sq = mult_partials_sum(derr_do, do_do_sq, o)
	derr_dshift_out = mult_partials_sum(derr_do, do_dshift_out, o)
	
	
	#return derr_do_sq[i_ind,j_ind]
	return derr_dshift_out[i_ind,j_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	#ref = o_sq
	ref = shift_out
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
