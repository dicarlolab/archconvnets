import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

layer_in = np.random.normal(size=(8,1))
w = np.random.normal(size=(9,8))

t = np.random.normal(size=(9,8))

#############
# relu below 'thresh' (0 = normal relu)
def relu(layer_in, thresh=0):
	temp = copy.deepcopy(layer_in)
	temp[layer_in < thresh] = thresh
	return temp

def relu_dlayer_in(layer_in, thresh=0):
	temp = np.ones_like(layer_in)
	temp[layer_in <= thresh] = 0
	
	temp2 = np.zeros(np.concatenate((layer_in.shape, layer_in.shape)))
	temp2[range(layer_in.shape[0]),:,range(layer_in.shape[0])] = temp[:,np.newaxis]
	
	return temp2

def f(y):
	#layer_in[i_ind] = y
	w[i_ind,j_ind] = y
	
	o = linear_F(w, layer_in)
	orelu = relu(o)
	
	return ((orelu - t)**2).sum()


def g(y):
	#layer_in[i_ind] = y
	w[i_ind,j_ind] = y
	
	o = linear_F(w, layer_in)
	orelu = relu(o)
	
	##
	derr_dorelu = sq_points_dinput(orelu - t)
	dorelu_do = relu_dlayer_in(orelu)
	
	derr_do = mult_partials(derr_dorelu, dorelu_do, orelu)
	
	do_dw = linear_F_dF_nsum_g(w, layer_in)
	do_dlayer_in = linear_F_dx_nsum_g(w, layer_in)
	
	derr_dw = mult_partials_collapse(derr_do, do_dw, o)
	derr_dlayer_in = mult_partials_collapse(derr_do, do_dlayer_in, o)
	
	##
	
	return derr_dw[i_ind,j_ind]
	#return derr_dlayer_in[i_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	#ref = layer_in
	ref = w
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = .2+ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	#y = .2+ref[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
