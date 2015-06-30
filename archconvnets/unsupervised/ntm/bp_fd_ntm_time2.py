#from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *

C = 4
M = 5
n_in = 3

t = np.random.normal(size=(C,M))
o_previ = np.random.normal(size=(C,M))
g = np.random.normal(size=(C,1))
w = np.random.normal(size=(C,n_in))
x = np.random.normal(size=(n_in))
x2 = np.random.normal(size=(n_in))

do_dwi = np.zeros((C,M,n_in))

def f(y):
	w[i_ind,j_ind] = y
	o_prev = copy.deepcopy(o_previ)
	
	###
	g = np.dot(w,x)
	o = o_prev*g[:,np.newaxis]
	
	o_prev = copy.deepcopy(o)
	
	###
	g = np.dot(w,x2)
	o = o_prev*g[:,np.newaxis]
	
	return ((o - t)**2).sum()


def g(y):
	w[i_ind,j_ind] = y
	do_dw = copy.deepcopy(do_dwi)
	o_prev = copy.deepcopy(o_previ)
	
	###
	g = np.dot(w,x)
	o = o_prev*g[:,np.newaxis]
	
	do_dw = np.einsum(do_dw, [0,1,2], g, [0], [0,1,2]) # g * do_dw
	do_dw += np.einsum(o_prev, [0,1], x, [2], [0,1,2]) # x * o^(t-1)
	
	o_prev = copy.deepcopy(o)
	
	###
	g = np.dot(w,x2)
	o = o_prev*g[:,np.newaxis]
	
	do_dw = np.einsum(do_dw, [0,1,2], g, [0], [0,1,2]) # g * do_dw
	do_dw += np.einsum(o_prev, [0,1], x2, [2], [0,1,2]) # x * o^(t-1)
	
	dw = np.einsum(2*(o - t), [0,1], do_dw, [0,1,2], [0,2])
	
	return dw[i_ind,j_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e0


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = w
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
		
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

