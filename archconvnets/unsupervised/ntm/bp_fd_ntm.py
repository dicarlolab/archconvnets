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

w_interp = np.abs(np.random.random(size=(n_controllers, n_mem_slots)))
beta_weights = np.abs(np.random.normal(size=(n_controllers, n_in)))
gamma_weights = np.abs(np.random.normal(size=(n_controllers, n_in)))
interp_weights = np.abs(np.random.random(size=(n_controllers, n_in)))
shift_weights = np.abs(np.random.normal(size=(n_controllers*n_shifts, n_in)))
key_weights = np.abs(np.random.normal(size=(n_controllers*m_length, n_in)))

w_prev = np.abs(np.random.normal(size=(n_controllers, n_mem_slots)))
w_content = np.abs(np.random.normal(size=(n_controllers, n_mem_slots)))

mem = np.random.normal(size=(n_mem_slots, m_length))

x = np.random.normal(size=(n_in,1))

t = np.random.normal(size=(n_controllers, n_mem_slots))

keys = linear_F(key_weights, x).reshape((n_controllers, m_length))

def f(y):
	#beta_weights[i_ind] = y
	#key_weights[i_ind,j_ind] = y
	#mem[i_ind,j_ind] = y
	#shift_weights[i_ind,j_ind] = y
	#gamma_weights[i_ind,j_ind] = y
	interp_weights[i_ind,j_ind] = y
	
	########
	# forward:
	
	# content
	keys = linear_F(key_weights, x).reshape((n_controllers, m_length))
	beta_out = linear_F(beta_weights, x)
	
	keys_focused = focus_keys(keys, beta_out)
	w_content = cosine_sim(keys_focused, mem)
	w_content_smax = softmax(w_content)
	
	# interpolation
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate(w_content_smax, w_prev, interp_gate_out)
	
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
	#beta_weights[i_ind] = y
	#key_weights[i_ind,j_ind] = y
	#mem[i_ind,j_ind] = y
	#shift_weights[i_ind,j_ind] = y
	#gamma_weights[i_ind,j_ind] = y
	interp_weights[i_ind,j_ind] = y
	
	########
	# forward:
	
	# content
	keys = linear_F(key_weights, x).reshape((n_controllers, m_length))
	beta_out = linear_F(beta_weights, x)
	
	keys_focused = focus_keys(keys, beta_out)
	w_content = cosine_sim(keys_focused, mem)
	w_content_smax = softmax(w_content)
	
	# interpolation
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate(w_content_smax, w_prev, interp_gate_out)
	
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
	dw_interp_dinterp_gate_out = interpolate_dinterp_gate_out(w_content_smax, w_prev, dw_tilde_dw_interp) # interpolate
	
	dw_interp_dw_content_smax = interpolate_dw_content(interp_gate_out, dw_tilde_dw_interp) # interpolate
	
	# content
	dw_content_smax_dw_content = softmax_dlayer_in(w_content_smax, dw_interp_dw_content_smax) # softmax
	
	dw_content_dkeys_focused = cosine_sim_dkeys(keys_focused, mem, dw_content_smax_dw_content) # cosine
	dw_content_dmem = cosine_sim_dmem(keys_focused, mem, dw_content_smax_dw_content) # cosine
	
	dfocus_key_dkeys = focus_key_dkeys(beta_out, dw_content_dkeys_focused).ravel()[:,np.newaxis] # focus
	dfocus_key_dbeta_out = focus_key_dbeta_out(keys, dw_content_dkeys_focused)[:,np.newaxis] # focus
	
	##########
	# backward (vars):
	dgamma_out_dgamma_weights = linear_dF(x, dw_dgamma_out) # gamma
	dshift_out_dshift_weights = linear_dF(x, dshift_out_relu_dshift_out) # shift_weights
	dinterp_gate_out_dinterp_weights = linear_dF(x, dw_interp_dinterp_gate_out) # interp_weights
	dbeta_out_dbeta_weights = linear_dF(x, dfocus_key_dbeta_out) # beta_weights
	dkeys_dkey_weights = linear_dF(x, dfocus_key_dkeys) # key_weights
	
	return dinterp_gate_out_dinterp_weights[i_ind,j_ind]
	#return dgamma_out_dgamma_weights[i_ind,j_ind]
	#return dshift_out_dshift_weights[i_ind,j_ind]
	#return dw_content_dmem[i_ind,j_ind]
	#return dkeys_dkey_weights[i_ind,j_ind]
	#return dbeta_out_dbeta_weights[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = interp_weights
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
		
	#i_ind = np.random.randint(ref.shape[0])
	#y = -1e0*ref[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

