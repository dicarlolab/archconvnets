import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

w = np.random.normal(size=(4,9))
inputs = np.random.normal(size=(9, 1))

t = np.random.normal(size=(4, 1))

def f(y):
	#w[i_ind,j_ind] = y
	inputs[i_ind] = y
	
	o = linear_F_sigmoid(w, inputs)
	
	return ((o - t)**2).sum()


def g(y):
	#w[i_ind,j_ind] = y
	inputs[i_ind] = y
	
	o = linear_F_sigmoid(w, inputs)
	
	##
	derr_do = sq_points_dinput(o - t)
	
	do_dw = linear_F_sigmoid_dF_nsum_g(o, w, inputs)
	do_dinputs = linear_F_sigmoid_dx_nsum_g(o, w, inputs)
	
	derr_dw = mult_partials_collapse(derr_do, do_dw, o)
	derr_dinputs = mult_partials_collapse(derr_do, do_dinputs, o)
	
	##
	
	#return derr_dw[i_ind,j_ind]
	return derr_dinputs[i_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = inputs#w
	i_ind = np.random.randint(ref.shape[0])
	#j_ind = np.random.randint(ref.shape[1])
	#y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	y = -1e0*ref[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
