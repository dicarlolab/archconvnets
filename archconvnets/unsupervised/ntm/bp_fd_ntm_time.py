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
n2 = 6
n1 = 7
n_in = 3

t = np.random.normal(size=(C,M))
o_previ = np.random.normal(size=(C,M))
g = np.random.normal(size=(C,1))

w3 = np.random.normal(size=(C,n2))
w2 = np.random.normal(size=(n2,n1))
w1 = np.random.normal(size=(n1,n_in))

x = np.random.normal(size=(n_in,1))
x2 = np.random.normal(size=(n_in,1))

do_dw3i = np.zeros((C,M,n2))
do_dw2i = np.zeros((C,M,n2,n1))
do_dw1i = np.zeros((C,M,n1,n_in))

################# interpolate simplified
def interpolate_simp(w_prev, interp_gate_out):
	return w_prev * interp_gate_out


def f(y):
	w1[i_ind,j_ind] = y
	
	o_prev = copy.deepcopy(o_previ)
	
	###
	g1 = sq_F(w1,x)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o = interpolate_simp(o_prev, g3)
	
	o_prev = copy.deepcopy(o)
	
	###
	g1 = sq_F(w1,x2)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o = interpolate_simp(o_prev, g3)
	
	o_prev = copy.deepcopy(o)
	
	return ((o - t)**2).sum()


def g(y):
	w1[i_ind,j_ind] = y
	
	do_dw3 = copy.deepcopy(do_dw3i)
	do_dw2 = copy.deepcopy(do_dw2i)
	do_dw1 = copy.deepcopy(do_dw1i)
	o_prev = copy.deepcopy(o_previ)
	
	###
	g1 = sq_F(w1,x)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o = interpolate_simp(o_prev, g3)
	
	# w1:
	dg3_dg2 = 2 * w3 * linear_F(w3,g2)
	dg2_dg1 = 2 * w2 * linear_F(w2,g1)
	dg1_dw1 = 2 * x.T * linear_F(w1,x)
	
	dg3_dg1 = np.einsum(dg3_dg2, [0,1], dg2_dg1, [1,2], [0,1,2])
	dg3_dw1 = np.einsum(dg3_dg1, [0,1,2], dg1_dw1, [2,3], [0,1,2,3])
	
	do_dw1 = np.einsum(do_dw1, range(4), g3, [0,4], range(4))
	do_dw1 += np.einsum(o_prev, [0,1], dg3_dw1, [0,4,2,3], range(4))
	
	# w2:
	dg2_dw2 = 2 * g1.T * linear_F(w2,g1)
	
	dg3_dw2 = np.einsum(dg3_dg2,[0,1], dg2_dw2, [1,2], [0,1,2])
	
	do_dw2 = np.einsum(do_dw2, range(4), g3, [0,3], range(4))
	do_dw2 += np.einsum(o_prev, [0,1], dg3_dw2, [0,2,3], range(4))
	
	# w3:
	dg3_dw3 = 2 * g2.T * linear_F(w3,g2)
	
	do_dw3 = np.einsum(do_dw3, [0,1,2], g3, [0,3], [0,1,2])
	do_dw3 += np.einsum(o_prev, [0,1], dg3_dw3, [0,2], [0,1,2])
	
	o_prev = copy.deepcopy(o)
	
	###
	g1 = sq_F(w1,x2)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o = interpolate_simp(o_prev, g3)
	
	# w1:
	dg3_dg2 = 2 * w3 * linear_F(w3,g2)
	dg2_dg1 = 2 * w2 * linear_F(w2,g1)
	dg1_dw1 = 2 * x2.T * linear_F(w1,x2)
	
	dg3_dg1 = np.einsum(dg3_dg2, [0,1], dg2_dg1, [1,2], [0,1,2])
	dg3_dw1 = np.einsum(dg3_dg1, [0,1,2], dg1_dw1, [2,3], [0,1,2,3])
	
	do_dw1 = np.einsum(do_dw1, range(4), g3, [0,4], range(4))
	do_dw1 += np.einsum(o_prev, [0,1], dg3_dw1, [0,4,2,3], range(4))
	
	# w2:
	dg2_dw2 = 2 * g1.T * linear_F(w2,g1)
	
	dg3_dw2 = np.einsum(dg3_dg2,[0,1], dg2_dw2, [1,2], [0,1,2])
	
	do_dw2 = np.einsum(do_dw2, range(4), g3, [0,3], range(4))
	do_dw2 += np.einsum(o_prev, [0,1], dg3_dw2, [0,2,3], range(4))
	
	# w3:
	dg3_dw3 = 2 * g2.T * linear_F(w3,g2)
	
	do_dw3 = np.einsum(do_dw3, [0,1,2], g3, [0,3], [0,1,2])
	do_dw3 += np.einsum(o_prev, [0,1], dg3_dw3, [0,2], [0,1,2])
	
	o_prev = copy.deepcopy(o)
	
	###
	
	dw1 = np.einsum(2*(o - t), [0,1], do_dw1, range(4), [2,3])
	dw2 = np.einsum(2*(o - t), [0,1], do_dw2, range(4), [2,3])
	dw3 = np.einsum(2*(o - t), [0,1], do_dw3, [0,1,2], [0,2])
	
	return dw1[i_ind,j_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e3


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = w1
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
		
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

