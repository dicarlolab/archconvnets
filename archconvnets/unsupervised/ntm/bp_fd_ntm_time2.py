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

SCALE = .5

t = np.random.normal(size=(C,M))
o_previ = np.random.normal(size=(C,M))
o_content = np.random.normal(size=(C,M))

w3 = np.random.normal(size=(C,n2)) * SCALE
w2 = np.random.normal(size=(n2,n1)) * SCALE
w1 = np.random.normal(size=(n1,n_in)) * SCALE

x = np.random.normal(size=(n_in,1))
x2 = np.random.normal(size=(n_in,1))

do_dw3i = np.zeros((C,M,n2))
do_dw2i = np.zeros((C,M,n2,n1))
do_dw1i = np.zeros((C,M,n1,n_in))

do_content_dw3 = np.zeros_like(do_dw3i)
do_content_dw2 = np.zeros_like(do_dw2i)
do_content_dw1 = np.zeros_like(do_dw1i)

def sq_points(input):
	return input**2

def sq_points_dinput(input):
	dinput = np.zeros((input.shape[0], input.shape[1], input.shape[0], input.shape[1]))
	for i in range(input.shape[0]):
		for j in range(input.shape[1]):
			dinput[i,j,i,j] = 2*input[i,j]
	return dinput

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
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o = sq_points(o_in)
	
	o_prev = copy.deepcopy(o)
	
	###
	g1 = sq_F(w1,x2)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o = sq_points(o_in)
	
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
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o = sq_points(o_in)
	
	do_do_in = sq_points_dinput(o_in)
	# w1:
	dg3_dg2 = sq_dlayer_in_nsum(w3, g2)
	dg2_dg1 = sq_dlayer_in_nsum(w2, g1)
	dg1_dw1 = sq_dF(w1, x, g1)
	
	dg3_dg1 = np.einsum(dg3_dg2, [0,1], dg2_dg1, [1,2], [0,1,2])
	dg3_dw1 = np.einsum(dg3_dg1, [0,1,2], dg1_dw1, [2,3], range(4))
	
	do_in_dw1 = np.einsum(do_dw1 + do_content_dw1, range(4), g3, [0,4], range(4))
	do_in_dw1 += np.einsum(o_prev + o_content, [0,1], dg3_dw1, [0,4,2,3], range(4))
	
	do_dw1 = np.einsum(do_in_dw1, [0,1,2,5], do_do_in, [3,4,0,1], [3,4,2,5])
	
	# w2:
	dg2_dw2 = sq_dF(w2, g1, g2)
	
	dg3_dw2 = np.einsum(dg3_dg2,[0,1], dg2_dw2, [1,2], [0,1,2])
	
	do_in_dw2 = np.einsum(do_dw2 + do_content_dw2, range(4), g3, [0,3], range(4))
	do_in_dw2 += np.einsum(o_prev + o_content, [0,1], dg3_dw2, [0,2,3], range(4))
	
	do_dw2 = np.einsum(do_in_dw2, [0,1,2,5], do_do_in, [3,4,0,1], [3,4,2,5])
	
	# w3:
	dg3_dw3 = sq_dF(w3, g2, g3)
	
	do_in_dw3 = np.einsum(do_dw3 + do_content_dw3, [0,1,2], g3, [0,3], [0,1,2])
	do_in_dw3 += np.einsum(o_prev + o_content, [0,1], dg3_dw3, [0,2], [0,1,2])
	
	do_dw3 = np.einsum(do_in_dw3, [0,1,2], do_do_in, [3,4,0,1], [3,4,2])
	
	o_prev = copy.deepcopy(o)
	
	###
	
	###
	g1 = sq_F(w1,x2)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o = sq_points(o_in)
	
	do_do_in = sq_points_dinput(o_in)
	# w1:
	dg3_dg2 = sq_dlayer_in_nsum(w3, g2)
	dg2_dg1 = sq_dlayer_in_nsum(w2, g1)
	dg1_dw1 = sq_dF(w1, x2, g1)
	
	dg3_dg1 = np.einsum(dg3_dg2, [0,1], dg2_dg1, [1,2], [0,1,2])
	dg3_dw1 = np.einsum(dg3_dg1, [0,1,2], dg1_dw1, [2,3], range(4))
	
	do_in_dw1 = np.einsum(do_dw1 + do_content_dw1, range(4), g3, [0,4], range(4))
	do_in_dw1 += np.einsum(o_prev + o_content, [0,1], dg3_dw1, [0,4,2,3], range(4))
	
	do_dw1 = np.einsum(do_in_dw1, [0,1,2,5], do_do_in, [3,4,0,1], [3,4,2,5])
	
	# w2:
	dg2_dw2 = sq_dF(w2, g1, g2)
	
	dg3_dw2 = np.einsum(dg3_dg2,[0,1], dg2_dw2, [1,2], [0,1,2])
	
	do_in_dw2 = np.einsum(do_dw2 + do_content_dw2, range(4), g3, [0,3], range(4))
	do_in_dw2 += np.einsum(o_prev + o_content, [0,1], dg3_dw2, [0,2,3], range(4))
	
	do_dw2 = np.einsum(do_in_dw2, [0,1,2,5], do_do_in, [3,4,0,1], [3,4,2,5])
	
	# w3:
	dg3_dw3 = sq_dF(w3, g2, g3)
	
	do_in_dw3 = np.einsum(do_dw3 + do_content_dw3, [0,1,2], g3, [0,3], [0,1,2])
	do_in_dw3 += np.einsum(o_prev + o_content, [0,1], dg3_dw3, [0,2], [0,1,2])
	
	do_dw3 = np.einsum(do_in_dw3, [0,1,2], do_do_in, [3,4,0,1], [3,4,2])
	
	o_prev = copy.deepcopy(o)
	
	###
	
	dw1 = np.einsum(2*(o - t), [0,1], do_dw1, range(4), [2,3])
	dw2 = np.einsum(2*(o - t), [0,1], do_dw2, range(4), [2,3])
	dw3 = np.einsum(2*(o - t), [0,1], do_dw3, [0,1,2], [0,2])
	
	return dw1[i_ind,j_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2


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

