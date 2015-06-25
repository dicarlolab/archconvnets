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

gamma_out = np.random.random(size=(n_controllers,1))

shift_weights = np.random.random(size=(n_controllers*n_shifts, n_in))
w_prev = np.random.random(size=(n_controllers, n_mem_slots))

x = np.random.random(size=(n_in,1))

t = np.random.random(size=(n_controllers, n_mem_slots))

############# sharpen across mem_slots separately for each controller
def sharpen(layer_in, gamma_out):
	# layer_in: [n_controllers, n_mem_slots], gamma_out: [n_controllers, 1]
	layer_in_raised = layer_in**gamma_out
	return layer_in_raised / layer_in_raised.sum(1)[:,np.newaxis]

def sharpen_dlayer_in(layer_in, gamma_out, above_w=1):
	# layer_in, above_w: [n_controllers, n_mem_slots], gamma_out: [n_controllers, 1]
	layer_in_raised = layer_in**gamma_out
	layer_in_raised_m1 = layer_in**(gamma_out-1)
	denom = layer_in_raised.sum(1)[:,np.newaxis] # sum across slots
	denom2 = denom**2
	
	# dsharpen[:,i]/dlayer_in[:,j] when i = j:
	dsharpen_dlayer_in = above_w*(layer_in_raised_m1 * denom - layer_in_raised * layer_in_raised_m1)

	# dsharpen[:,i]/dlayer_in[:,j] when i != j:
	dsharpen_dlayer_in -=  np.einsum(layer_in_raised_m1, [0,1], above_w * layer_in_raised, [0,2], [0,1])
	dsharpen_dlayer_in += above_w * layer_in_raised_m1 * layer_in_raised
	
	dsharpen_dlayer_in *= gamma_out / denom2
	
	return dsharpen_dlayer_in

def sharpen_dgamma_out(layer_in, gamma_out, above_w=1):
	# layer_in, above_w: [n_controllers, n_mem_slots], gamma_out: [n_controllers, 1]
	w_gamma = layer_in**gamma_out
	w_gamma_sum = w_gamma.sum(1)[:,np.newaxis] # sum across slots
	
	ln_gamma = np.log(layer_in)
	
	dw_gamma_dgamma = w_gamma * ln_gamma
	dw_sum_gamma_dgamma = dw_gamma_dgamma.sum(1)[:,np.newaxis] # sum across slots
	
	dw_dgamma = (dw_gamma_dgamma * w_gamma_sum - w_gamma * dw_sum_gamma_dgamma) / (w_gamma_sum**2)
	return (dw_dgamma * above_w).sum(1)[:,np.newaxis]

############
def softmax(layer_in):
	# layer_in: [n_in]
	return np.exp(layer_in)/np.sum(np.exp(layer_in))

def softmax_dlayer_in(layer_out, above_w=1):
	# layer_in, above_w: [n_in]
	
	# dsoftmax[i]/dlayer_in[j] when i = j:
	temp = above_w * (layer_out * (1 - layer_out))
	
	# dsoftmax[i]/dlayer_in[j] when i != j:
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

# todo: shift_w_dw_prev()

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
	#shift_weights[i_ind,j_ind] = y
	gamma_out[i_ind] = y
	#x[i_ind] = y
	
	########
	# forward:
	shift_out = linear_F(shift_weights, x).reshape((n_controllers, n_shifts))
	shift_out_smax = softmax(shift_out.ravel()).reshape((n_controllers, n_shifts))
	w_tilde = shift_w(shift_out_smax, w_prev)
	w = sharpen(w_tilde, gamma_out)
	
	return ((w - t)**2).sum()

def g(y):
	#shift_weights[i_ind,j_ind] = y
	gamma_out[i_ind] = y
	#x[i_ind] = y
	
	########
	# forward:
	shift_out = linear_F(shift_weights, x).reshape((n_controllers, n_shifts))
	shift_out_smax = softmax(shift_out.ravel()).reshape((n_controllers, n_shifts))
	w_tilde = shift_w(shift_out_smax, w_prev)
	w = sharpen(w_tilde, gamma_out)
	
	##########
	# backward (data):
	dsharpen_dw_tilde = sharpen_dlayer_in(w_tilde, gamma_out, 2*(w - t))
	dw_tilde_dshift_out = shift_w_dshift_out(w_prev, dsharpen_dw_tilde).reshape((n_controllers*n_shifts,1))
	dshift_out_smax_dshift_out = softmax_dlayer_in(shift_out_smax.ravel()[:,np.newaxis], dw_tilde_dshift_out)
	dshift_out_dx = linear_dlayer_in(shift_weights, dshift_out_smax_dshift_out) ##
	
	# backward (vars):
	dw_dgamma_out = sharpen_dgamma_out(w_tilde, gamma_out, 2*(w - t))
	dshift_out_dshift_weights = linear_dF(x, dshift_out_smax_dshift_out)
	
	return dw_dgamma_out[i_ind]
	#return dshift_out_dx[i_ind]
	#return dshift_out_dshift_weights[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2#9#10#10


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	'''i_ind = np.random.randint(shift_weights.shape[0])
	j_ind = np.random.randint(shift_weights.shape[1])
	y = -1e0*shift_weights[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);'''
	
	#i_ind = np.random.randint(x.shape[0])
	#y = -1e0*x[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);
	
	i_ind = np.random.randint(gamma_out.shape[0])
	y = -1e0*gamma_out[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);
	
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

