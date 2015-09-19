import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

w = np.abs(np.random.normal(size=(8,1)))
inputs = np.abs(np.random.normal(size=(8, 15)))

t = np.random.normal(size=(8, 15))

def f(y):
	#inputs[i_ind,j_ind] = y
	w[i_ind] = y
	
	o = sharpen(inputs, w)
	
	return ((o - t)**2).sum()


def g(y):
	#inputs[i_ind,j_ind] = y
	w[i_ind] = y
	
	o = sharpen(w, inputs)
	
	##
	derr_do = sq_points_dinput(o - t)
	
	do_dw = sharpen_dgamma_out_nsum(inputs, w)
	
	derr_dw = mult_partials_collapse(derr_do, do_dw, o)
	
	#print do_dw.shape, derr_dw.shape, o.shape
	##
	
	return derr_dw[i_ind]
	#return derr_dinputs[i_ind,j_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e7

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = w
	i_ind = np.random.randint(ref.shape[0])
	#j_ind = np.random.randint(ref.shape[1])
	#y = .2+ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	y = .1 + ref[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)[0]
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
