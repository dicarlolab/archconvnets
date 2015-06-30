#from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *

n_out = 7
n_in = 5
n_shifts = 3 # must be 3 [shift_w()]
n_controllers = 4
n_mem_slots = 2 # "M"
m_length = 7 # "N"

SCALE = 1e-1

beta_weights = np.random.normal(size=(n_controllers, n_in)) * SCALE
gamma_weights = np.random.normal(size=(n_controllers, n_in)) * SCALE
interp_weights = np.random.normal(size=(n_controllers, n_in)) * SCALE
shift_weights = np.random.normal(size=(n_controllers*n_shifts, n_in)) * SCALE
key_weights = np.random.normal(size=(n_controllers*m_length, n_in)) * SCALE
out_weights = np.random.normal(size=(n_out, n_controllers*m_length)) * SCALE
out_bypass_weights = np.random.normal(size=(n_out, n_in)) * SCALE

beta_biases = 1 + np.random.normal(size=(n_controllers, 1)) * SCALE
gamma_biases = np.random.normal(size=(n_controllers, 1)) * SCALE
interp_biases = np.random.normal(size=(n_controllers, 1)) * SCALE
shift_biases = np.random.normal(size=(n_controllers*n_shifts, 1)) * SCALE
key_biases = np.random.normal(size=(n_controllers*m_length, 1)) * SCALE
out_biases = np.random.normal(size=(n_out, 1)) * SCALE

w_prev = np.abs(np.random.normal(size=(n_controllers, n_mem_slots))) * SCALE

mem = np.random.normal(size=(n_mem_slots, m_length)) * SCALE

x = np.random.normal(size=(n_in,1)) * SCALE

t = np.random.normal(size=(n_out, 1)) * SCALE

def f(y):
	mem[i_ind,j_ind] = y
	
	#beta_weights[i_ind] = y
	#key_weights[i_ind,j_ind] = y
	#shift_weights[i_ind,j_ind] = y
	#gamma_weights[i_ind,j_ind] = y
	#interp_weights[i_ind,j_ind] = y
	#out_weights[i_ind,j_ind] = y
	#out_bypass_weights[i_ind,j_ind] = y
	
	#gamma_biases[i_ind] = y
	#shift_biases[i_ind] = y
	#key_biases[i_ind] = y
	#interp_biases[i_ind] = y
	#out_biases[i_ind] = y
	
	########
	# forward:
	
	# content
	keys = (linear_F(key_weights, x) + key_biases).reshape((n_controllers, m_length))
	keys_relu = relu(keys)
	
	beta_out = linear_F(beta_weights, x) + beta_biases ##?
	beta_out_relu = relu(beta_out)
	
	keys_focused = focus_keys(keys_relu, beta_out_relu)
	w_content = cosine_sim(keys_focused, mem)
	w_content_smax = softmax(w_content)
	
	# interpolation
	interp_gate_out = linear_F(interp_weights, x) + interp_biases
	interp_gate_out_sigm = sigmoid(interp_gate_out)
	w_interp = interpolate(w_content_smax, w_prev, interp_gate_out_sigm)
	
	# shift
	shift_out = linear_F(shift_weights, x) + shift_biases
	shift_out_relu = relu(shift_out)
	
	shift_out_relu_smax = softmax(shift_out_relu.reshape((n_controllers, n_shifts)))
	w_tilde = shift_w(shift_out_relu_smax, w_interp)
	
	# sharpen
	gamma_out = linear_F(gamma_weights, x) + gamma_biases
	gamma_out_relu = relu(gamma_out,1)
	
	w = sharpen(w_tilde, gamma_out_relu)
	
	# read mem
	read_mem = read_from_mem(w, mem)
	
	# output layer
	out = linear_F(out_weights, read_mem.ravel()[:,np.newaxis]) + out_biases
	out += linear_F(out_bypass_weights, x)
	
	return ((out - t)**2).sum()

def g(y):
	mem[i_ind,j_ind] = y
	
	#beta_weights[i_ind] = y
	#key_weights[i_ind,j_ind] = y
	#shift_weights[i_ind,j_ind] = y
	#gamma_weights[i_ind,j_ind] = y
	#interp_weights[i_ind,j_ind] = y
	#out_weights[i_ind,j_ind] = y
	#out_bypass_weights[i_ind,j_ind] = y
	
	#gamma_biases[i_ind] = y
	#shift_biases[i_ind] = y
	#key_biases[i_ind] = y
	#interp_biases[i_ind] = y
	#out_biases[i_ind] = y
	
	########
	# forward:
	
	# content
	keys = (linear_F(key_weights, x) + key_biases).reshape((n_controllers, m_length))
	keys_relu = relu(keys)
	
	beta_out = linear_F(beta_weights, x) + beta_biases ##?
	beta_out_relu = relu(beta_out)
	
	keys_focused = focus_keys(keys_relu, beta_out_relu)
	w_content = cosine_sim(keys_focused, mem)
	w_content_smax = softmax(w_content)
	
	# interpolation
	interp_gate_out = linear_F(interp_weights, x) + interp_biases
	interp_gate_out_sigm = sigmoid(interp_gate_out)
	w_interp = interpolate(w_content_smax, w_prev, interp_gate_out_sigm)
	
	# shift
	shift_out = linear_F(shift_weights, x) + shift_biases
	shift_out_relu = relu(shift_out)
	
	shift_out_relu_smax = softmax(shift_out_relu.reshape((n_controllers, n_shifts)))
	w_tilde = shift_w(shift_out_relu_smax, w_interp)
	
	# sharpen
	gamma_out = linear_F(gamma_weights, x) + gamma_biases
	gamma_out_relu = relu(gamma_out,1)
	
	w = sharpen(w_tilde, gamma_out_relu)
	
	# read mem
	read_mem = read_from_mem(w, mem)
	
	# output layer
	out = linear_F(out_weights, read_mem.ravel()[:,np.newaxis]) + out_biases
	out += linear_F(out_bypass_weights, x)
	
	##########
	# backward (data):
	
	# output layer
	dout_dread_mem = linear_dlayer_in(out_weights, 2*(out - t)).reshape((n_controllers, m_length))
	
	# read mem
	dread_from_mem_dw = read_from_mem_dw(mem, dout_dread_mem) # read mem, dw
	dread_from_mem_dmem = read_from_mem_dmem(w, dout_dread_mem) # read mem, dmem
	
	# sharpen
	dsharpen_dw_tilde = sharpen_dlayer_in(w_tilde, gamma_out_relu, dread_from_mem_dw) # sharpen
	dw_dgamma_out_relu = sharpen_dgamma_out(w_tilde, gamma_out_relu, dread_from_mem_dw) # sharpen
	dw_dgamma_out = relu_dlayer_in(gamma_out, dw_dgamma_out_relu, 1) # relu
	
	# shift
	dw_tilde_dshift_out = shift_w_dshift_out(w_interp, dsharpen_dw_tilde) # shift
	dshift_out_relu_smax_dshift_out = softmax_dlayer_in(shift_out_relu_smax, dw_tilde_dshift_out) # softmax
	dshift_out_relu_dshift_out = relu_dlayer_in(shift_out, dshift_out_relu_smax_dshift_out.ravel()[:,np.newaxis]) # relu

	# interpolation
	dw_tilde_dw_interp = shift_w_dw_interp(shift_out_relu_smax, dsharpen_dw_tilde) # shift
	dw_interp_dw_content_smax = interpolate_dw_content(interp_gate_out_sigm, dw_tilde_dw_interp) # interpolate
	
	dw_interp_dinterp_gate_out_sigm = interpolate_dinterp_gate_out(w_content_smax, w_prev, dw_tilde_dw_interp) # interpolate
	dinterp_gate_out_sigm_dinterp_gate_out = sigmoid_dlayer_in(interp_gate_out_sigm, dw_interp_dinterp_gate_out_sigm) # sigmoid
	
	# content
	dw_content_smax_dw_content = softmax_dlayer_in(w_content_smax, dw_interp_dw_content_smax) # softmax
	
	dw_content_dkeys_focused = cosine_sim_dkeys(keys_focused, mem, dw_content_smax_dw_content) # cosine
	dw_content_dmem = cosine_sim_dmem(keys_focused, mem, dw_content_smax_dw_content) # cosine
	
	dfocus_key_dbeta_out_relu = focus_key_dbeta_out(keys_relu, dw_content_dkeys_focused) # focus, dbeta
	dbeta_out_relu_dbeta_out = relu_dlayer_in(beta_out_relu, dfocus_key_dbeta_out_relu).ravel()[:,np.newaxis] # relu, dbeta
	
	dfocus_key_dkeys_relu = focus_key_dkeys(beta_out_relu, dw_content_dkeys_focused) # focus, dkeys
	dkeys_relu_dkeys = relu_dlayer_in(keys, dfocus_key_dkeys_relu).ravel()[:,np.newaxis] # relu, dkeys
	
	
	##########
	# backward (vars):
	dgamma_out_dgamma_weights = linear_dF(x, dw_dgamma_out) # gamma
	dshift_out_dshift_weights = linear_dF(x, dshift_out_relu_dshift_out) # shift_weights
	dinterp_gate_out_dinterp_weights = linear_dF(x, dinterp_gate_out_sigm_dinterp_gate_out) # interp_weights
	dbeta_out_dbeta_weights = linear_dF(x, dbeta_out_relu_dbeta_out) # beta_weights
	dkeys_dkey_weights = linear_dF(x, dkeys_relu_dkeys) # key_weights
	dout_dout_weights = linear_dF(read_mem.ravel()[:,np.newaxis], 2*(out - t)) # out_weights
	dout_dout_bypass_weights = linear_dF(x, 2*(out - t)) # out_bypass_weights
	
	return dw_content_dmem[i_ind,j_ind] + dread_from_mem_dmem[i_ind,j_ind]
	
	#return dw_dgamma_out[i_ind] # gamma_biases
	#return dshift_out_relu_dshift_out[i_ind] # shift_biases
	#return dinterp_gate_out_sigm_dinterp_gate_out[i_ind] # interp_biases
	#return dbeta_out_relu_dbeta_out[i_ind] # beta_biases
	#return dkeys_relu_dkeys[i_ind] # key_biases
	#return (2*(out - t))[i_ind]
	
	#return dinterp_gate_out_dinterp_weights[i_ind,j_ind]
	#return dgamma_out_dgamma_weights[i_ind,j_ind]
	#return dshift_out_dshift_weights[i_ind,j_ind]
	#return dbeta_out_dbeta_weights[i_ind,j_ind]
	#return dkeys_dkey_weights[i_ind,j_ind]
	#return dout_dout_weights[i_ind,j_ind]
	#return dout_dout_bypass_weights[i_ind,j_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e0


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = mem
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

