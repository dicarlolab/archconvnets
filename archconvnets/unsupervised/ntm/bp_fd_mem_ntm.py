from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy

n_in = 4
n_shifts = 3 # must be 3 [shift_w()]
n_controllers = 2
n_mem_slots = 5 # "M"
m_length = 8 # "N"

shift_weights = np.random.random(size=(n_controllers*n_shifts, n_in))
w_prev = np.random.random(size=(n_controllers, n_mem_slots))

x = np.random.random(size=(n_in,1))

t = np.random.random(size=(n_controllers, n_mem_slots))

def softmax(layer_in):
	return np.exp(layer_in)/np.sum(np.exp(layer_in))

def softmax_dlayer_in(layer_out, above_w=1):
	temp = above_w * (layer_out * (1 - layer_out))
	temp += -np.einsum(np.squeeze(above_w*layer_out), [0], np.squeeze(layer_out), [1], [1])[:,np.newaxis] + above_w*layer_out**2
	return temp

############### shift w
def shift_w(shift_out, w_prev):	
	# shift_out: [n_controllers, n_shifts], w_prev: [n_controllers, mem_length]
	
	w_tilde = np.zeros_like(w_prev)
	
	for loc in range(n_mem_slots):
		w_tilde[:,loc] = shift_out[:,0]*w_prev[:,loc-1] + shift_out[:,1]*w_prev[:,loc] + \
				shift_out[:,2]*w_prev[:,(loc+1)%n_mem_slots]
	return w_tilde # [n_controllers, mem_length]

def shift_w_dshift_out(w_prev, above_w=1):
	# w_prev: [n_controllers, mem_length], above_w: [n_controllers, mem_length]
	
	dshift_w_dshift_out = np.zeros((n_controllers, n_mem_slots, n_shifts))
	for m in range(n_mem_slots):
		for H in [-1,0,1]:
			dshift_w_dshift_out[:,m,H+1] = w_prev[:, (m+H)%n_mem_slots] * above_w[:,m]
	
	return dshift_w_dshift_out.sum(1) # [n_controllers, n_shifts]

############## linear layer
def linear_F(F, layer_in):
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in) # [n1, 1]

def linear_dF(layer_in, above_w): 
	# layer_in: [n_in, 1], above_w: [n1,1]
	
	return layer_in.T*above_w # [n1, n_in]

def linear_dlayer_in(F, above_w=1):
	return (F*above_w).sum(0)[:,np.newaxis] # [n_in, 1]

################## squared layer
def sq_F(F, layer_in):
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in)**2 # [n1, 1]

def sq_dF(F, layer_in, layer_out, above_w): 
	# F: [n1, n_in], layer_in: [n_in, 1], layer_out,above_w: [n1,1]
	
	s = np.sign(np.dot(F,layer_in))
	return 2*s*np.sqrt(layer_out)*(layer_in.T)*above_w # [n1, n_in]

def sq_dlayer_in(F, layer_in, layer_out, above_w=1):
	s = np.sign(np.dot(F,layer_in))
	return 2*(s*np.sqrt(layer_out)*F*above_w).sum(0)[:,np.newaxis] # [n_in, 1]
	
i_ind = 1
j_ind = 1


def f(y):
	shift_weights[i_ind,j_ind] = y
	#x[i_ind] = y
	
	shift_out = linear_F(shift_weights, x).reshape((n_controllers, n_shifts))
	shift_out_smax = softmax(shift_out.ravel()).reshape((n_controllers, n_shifts))
	w_tilde = shift_w(shift_out_smax, w_prev)
	
	return ((w_tilde - t)**2).sum()

def g(y):
	shift_weights[i_ind,j_ind] = y
	#x[i_ind] = y
	
	shift_out = linear_F(shift_weights, x).reshape((n_controllers, n_shifts))
	shift_out_smax = softmax(shift_out.ravel()).reshape((n_controllers, n_shifts))
	w_tilde = shift_w(shift_out_smax, w_prev)
	
	dw_tilde_dshift_out = shift_w_dshift_out(w_prev, 2*(w_tilde - t)).reshape((n_controllers*n_shifts,1))
	
	dshift_out_smax_dshift_out = softmax_dlayer_in(shift_out_smax.ravel()[:,np.newaxis], dw_tilde_dshift_out)
	
	dshift_out_dshift_weights = linear_dF(x, dshift_out_smax_dshift_out)
	dshift_out_dx = linear_dlayer_in(shift_weights, dshift_out_smax_dshift_out)
	
	#return dshift_out_dx[i_ind]
	return dshift_out_dshift_weights[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2#9#10#10


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	i_ind = np.random.randint(shift_weights.shape[0])
	j_ind = np.random.randint(shift_weights.shape[1])
	y = -1e0*shift_weights[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);
	
	#i_ind = np.random.randint(x.shape[0])
	#y = -1e0*x[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);
	
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

