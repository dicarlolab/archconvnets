#from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *

n_in = 4
n_shifts = 3 # must be 3 [shift_w()]
n_controllers = 2
n_mem_slots = 5 # "M"
m_length = 8 # "N"

w_interp = np.random.random(size=(n_controllers, n_mem_slots))
gamma_weights = np.random.normal(size=(n_controllers, n_in))
interp_weights = np.random.random(size=(n_controllers, n_in))
shift_weights = np.random.normal(size=(n_controllers*n_shifts, n_in),scale=1)

w_prev = np.abs(np.random.normal(size=(n_controllers, n_mem_slots)))
w_content = np.abs(np.random.normal(size=(n_controllers, n_mem_slots)))

keys = np.random.normal(size=(n_controllers, m_length))
mem = np.random.normal(size=(n_mem_slots, m_length))

x = np.random.normal(size=(n_in,1))

t = np.random.normal(size=(n_controllers, n_mem_slots))

############
# cosine similarity between each controller's key and memory vector
def cosine_sim(keys, mem):
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	
	return numer / denom # [n_controllers, n_mem_slots]

def cosine_sim_dkeys(keys, mem, above_w=1):
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	above_w_denom2 = above_w/(denom**2)
	
	denom_keys = np.sqrt(np.sum(keys**2,1))
	denom_mem = np.sqrt(np.sum(mem**2,1))
	
	dnumer_keys = np.dot(denom*above_w_denom2, mem)
	ddenom_keys = keys * (np.dot(numer*above_w_denom2, denom_mem)/denom_keys)[:,np.newaxis]

	return dnumer_keys - ddenom_keys # [n_controllers, m_length]


def f(y):
	#w_content[i_ind,j_ind] = y
	keys[i_ind,j_ind] = y
	#shift_weights[i_ind,j_ind] = y
	#gamma_weights[i_ind,j_ind] = y
	#interp_weights[i_ind,j_ind] = y
	
	########
	# forward:
	
	# content
	#w_content = cosine_sim(keys, mem)
	w_content = cosine_sim(keys, mem)
	
	# interpolation
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate(w_content, w_prev, interp_gate_out)
	
	# shift
	shift_out = linear_F(shift_weights, x)
	shift_out_relu = relu(shift_out)
	
	shift_out_relu_smax = softmax(shift_out_relu.reshape((n_controllers, n_shifts)))
	w_tilde = shift_w(shift_out_relu_smax, w_interp)
	
	# sharpen
	gamma_out = linear_F(gamma_weights, x)
	gamma_out_relu = relu(gamma_out,1)
	
	w = sharpen(w_tilde, gamma_out_relu)
	
	return ((w - t)**2).sum()

def g(y):
	keys[i_ind,j_ind] = y
	#shift_weights[i_ind,j_ind] = y
	#gamma_weights[i_ind,j_ind] = y
	#interp_weights[i_ind,j_ind] = y
	
	########
	# forward:
	
	# content
	w_content = cosine_sim(keys, mem)
	
	# interpolation
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate(w_content, w_prev, interp_gate_out)
	
	# shift
	shift_out = linear_F(shift_weights, x)
	shift_out_relu = relu(shift_out)
	
	shift_out_relu_smax = softmax(shift_out_relu.reshape((n_controllers, n_shifts)))
	w_tilde = shift_w(shift_out_relu_smax, w_interp)
	
	# sharpen
	gamma_out = linear_F(gamma_weights, x)
	gamma_out_relu = relu(gamma_out,1)
	
	w = sharpen(w_tilde, gamma_out_relu)
	
	##########
	# backward (data):
	
	# sharpen
	dsharpen_dw_tilde = sharpen_dlayer_in(w_tilde, gamma_out_relu, 2*(w - t)) # sharpen
	dw_dgamma_out_relu = sharpen_dgamma_out(w_tilde, gamma_out_relu, 2*(w - t)) # sharpen
	dw_dgamma_out = relu_dlayer_in(gamma_out, dw_dgamma_out_relu, 1) # relu
	
	# shift
	dw_tilde_dshift_out = shift_w_dshift_out(w_interp, dsharpen_dw_tilde) # shift
	dshift_out_relu_smax_dshift_out = softmax_dlayer_in(shift_out_relu_smax, dw_tilde_dshift_out) # softmax
	dshift_out_relu_dshift_out = relu_dlayer_in(shift_out, dshift_out_relu_smax_dshift_out.ravel()[:,np.newaxis]) # relu

	# interpolation
	dw_tilde_dw_interp = shift_w_dw_interp(shift_out_relu_smax, dsharpen_dw_tilde) # shift
	dw_interp_dinterp_gate_out = interpolate_dinterp_gate_out(w_content, w_prev, dw_tilde_dw_interp) # interpolate
	
	dw_interp_dw_content = interpolate_dw_content(interp_gate_out, dw_tilde_dw_interp) # interpolate
	
	# cosine
	dw_content_dkeys = cosine_sim_dkeys(keys, mem, dw_interp_dw_content)
	
	##########
	# backward (vars):
	dgamma_out_dgamma_weights = linear_dF(x, dw_dgamma_out)
	dshift_out_dshift_weights = linear_dF(x, dshift_out_relu_dshift_out)
	dinterp_gate_out_dinterp_weights = linear_dF(x, dw_interp_dinterp_gate_out)
	
	#return dinterp_gate_out_dinterp_weights[i_ind,j_ind]
	#return dgamma_out_dgamma_weights[i_ind,j_ind]
	#return dshift_out_dshift_weights[i_ind,j_ind]
	
	return dw_content_dkeys[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = keys#w_content
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
		
	#i_ind = np.random.randint(gamma_weights.shape[0])
	#y = -1e0*gamma_weights[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

