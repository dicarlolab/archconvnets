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

BETA = 0
GAMMA = 1
INTERP = 2
SHIFT = 3
KEY = 4

O_KEYS = 0
O_KEYS_RELU = 1
O_BETA_OUT = 2
O_BETA_OUT_RELU = 3
O_KEYS_FOCUSED = 4
O_W_CONTENT = 5
O_W_CONTENT_SMAX = 6
O_INTERP_GATE_OUT = 7
O_INTERP_GATE_OUT_SIGM = 8
O_W_INTERP = 9
O_SHIFT_OUT = 10
O_SHIFT_OUT_RELU = 11
O_SHIFT_OUT_RELU_SMAX = 12
O_W_TILDE = 13
O_GAMMA_OUT = 14
O_GAMMA_OUT_RELU = 15
O_W = 16

## head weights
beta_weights = np.random.normal(size=(n_controllers, n_in)) * SCALE
gamma_weights = np.random.normal(size=(n_controllers, n_in)) * SCALE
interp_weights = np.random.normal(size=(n_controllers, n_in)) * SCALE
shift_weights = np.random.normal(size=(n_controllers*n_shifts, n_in)) * SCALE
key_weights = np.random.normal(size=(n_controllers*m_length, n_in)) * SCALE

beta_biases = 1 + np.random.normal(size=(n_controllers, 1)) * SCALE
gamma_biases = np.random.normal(size=(n_controllers, 1)) * SCALE
interp_biases = np.random.normal(size=(n_controllers, 1)) * SCALE
shift_biases = np.random.normal(size=(n_controllers*n_shifts, 1)) * SCALE
key_biases = np.random.normal(size=(n_controllers*m_length, 1)) * SCALE

##
out_weights = np.random.normal(size=(n_out, n_controllers*m_length)) * SCALE
out_bypass_weights = np.random.normal(size=(n_out, n_in)) * SCALE

out_biases = np.random.normal(size=(n_out, 1)) * SCALE

w_prev = np.abs(np.random.normal(size=(n_controllers, n_mem_slots))) * SCALE

mem = np.random.normal(size=(n_mem_slots, m_length)) * SCALE

x = np.random.normal(size=(n_in,1)) * SCALE
t = np.random.normal(size=(n_out, 1)) * SCALE

W_READ = [beta_weights, gamma_weights, interp_weights, shift_weights, key_weights]
B_READ = [beta_biases, gamma_biases, interp_biases, shift_biases, key_biases]

W_WRITE = copy.deepcopy(W_READ)
B_WRITE = copy.deepcopy(B_READ)

REF = INTERP

def head_forward(x, W, B, w_prev, mem):
	# content
	keys = (linear_F(W[KEY], x) + B[KEY]).reshape((n_controllers, m_length))
	keys_relu = relu(keys)
	
	beta_out = linear_F(W[BETA], x) + B[BETA] ##?
	beta_out_relu = relu(beta_out)
	
	keys_focused = focus_keys(keys_relu, beta_out_relu)
	w_content = cosine_sim(keys_focused, mem)
	w_content_smax = softmax(w_content)
	
	# interpolation
	interp_gate_out = linear_F(W[INTERP], x) + B[INTERP]
	interp_gate_out_sigm = sigmoid(interp_gate_out)
	w_interp = interpolate(w_content_smax, w_prev, interp_gate_out_sigm)
	
	# shift
	shift_out = linear_F(W[SHIFT], x) + B[SHIFT]
	shift_out_relu = relu(shift_out)
	
	shift_out_relu_smax = softmax(shift_out_relu.reshape((n_controllers, n_shifts)))
	w_tilde = shift_w(shift_out_relu_smax, w_interp)
	
	# sharpen
	gamma_out = linear_F(W[GAMMA], x) + B[GAMMA]
	gamma_out_relu = relu(gamma_out,1)
	
	w = sharpen(w_tilde, gamma_out_relu)
	
	OUT = [keys, keys_relu, beta_out, beta_out_relu, keys_focused, w_content, w_content_smax, interp_gate_out, interp_gate_out_sigm, w_interp,\
			shift_out, shift_out_relu, shift_out_relu_smax, w_tilde, gamma_out, gamma_out_relu, w]
	
	return OUT
	
def head_backward(x, W, B, w_prev, mem, OUT, above_w):
	# sharpen
	dsharpen_dw_tilde = sharpen_dlayer_in(OUT[O_W_TILDE], OUT[O_GAMMA_OUT_RELU], above_w) # sharpen
	dw_dgamma_out_relu = sharpen_dgamma_out(OUT[O_W_TILDE], OUT[O_GAMMA_OUT_RELU], above_w) # sharpen
	dw_dgamma_out = relu_dlayer_in(OUT[O_GAMMA_OUT], dw_dgamma_out_relu, 1) # relu
	
	# shift
	dw_tilde_dshift_out = shift_w_dshift_out(OUT[O_W_INTERP], dsharpen_dw_tilde) # shift
	dshift_out_relu_smax_dshift_out = softmax_dlayer_in(OUT[O_SHIFT_OUT_RELU_SMAX], dw_tilde_dshift_out) # softmax
	dshift_out_relu_dshift_out = relu_dlayer_in(OUT[O_SHIFT_OUT], dshift_out_relu_smax_dshift_out.ravel()[:,np.newaxis]) # relu

	# interpolation
	dw_tilde_dw_interp = shift_w_dw_interp(OUT[O_SHIFT_OUT_RELU_SMAX], dsharpen_dw_tilde) # shift
	dw_interp_dw_content_smax = interpolate_dw_content(OUT[O_INTERP_GATE_OUT_SIGM], dw_tilde_dw_interp) # interpolate
	
	dw_interp_dinterp_gate_out_sigm = interpolate_dinterp_gate_out(OUT[O_W_CONTENT_SMAX], w_prev, dw_tilde_dw_interp) # interpolate ############
	dinterp_gate_out_sigm_dinterp_gate_out = sigmoid_dlayer_in(OUT[O_INTERP_GATE_OUT_SIGM], dw_interp_dinterp_gate_out_sigm) # sigmoid
	
	# content
	dw_content_smax_dw_content = softmax_dlayer_in(OUT[O_W_CONTENT_SMAX], dw_interp_dw_content_smax) # softmax
	
	dw_content_dkeys_focused = cosine_sim_dkeys(OUT[O_KEYS_FOCUSED], mem, dw_content_smax_dw_content) # cosine
	dw_content_dmem = cosine_sim_dmem(OUT[O_KEYS_FOCUSED], mem, dw_content_smax_dw_content) # cosine ########################
	
	dfocus_key_dbeta_out_relu = focus_key_dbeta_out(OUT[O_KEYS_RELU], dw_content_dkeys_focused) # focus, dbeta
	dbeta_out_relu_dbeta_out = relu_dlayer_in(OUT[O_BETA_OUT_RELU], dfocus_key_dbeta_out_relu).ravel()[:,np.newaxis] # relu, dbeta
	
	dfocus_key_dkeys_relu = focus_key_dkeys(OUT[O_BETA_OUT_RELU], dw_content_dkeys_focused) # focus, dkeys
	dkeys_relu_dkeys = relu_dlayer_in(OUT[O_KEYS], dfocus_key_dkeys_relu).ravel()[:,np.newaxis] # relu, dkeys
	
	##########
	# backward (filters):
	dgamma_out_dgamma_weights = linear_dF(x, dw_dgamma_out) # gamma
	dshift_out_dshift_weights = linear_dF(x, dshift_out_relu_dshift_out) # shift_weights
	dinterp_gate_out_dinterp_weights = linear_dF(x, dinterp_gate_out_sigm_dinterp_gate_out) # interp_weights
	dbeta_out_dbeta_weights = linear_dF(x, dbeta_out_relu_dbeta_out) # beta_weights
	dkeys_dkey_weights = linear_dF(x, dkeys_relu_dkeys) # key_weights
	
	DW = [dbeta_out_dbeta_weights, dgamma_out_dgamma_weights, dinterp_gate_out_dinterp_weights, \
			dshift_out_dshift_weights, dkeys_dkey_weights]
	DB = [dbeta_out_relu_dbeta_out, dw_dgamma_out, dinterp_gate_out_sigm_dinterp_gate_out, \
			dshift_out_relu_dshift_out, dkeys_relu_dkeys]
	
	return DW, DB, dw_content_dmem

def f(y):
	#mem[i_ind,j_ind] = y
	
	#out_weights[i_ind,j_ind] = y
	#out_bypass_weights[i_ind,j_ind] = y
	
	W_READ[REF][i_ind,j_ind] = y
	#B_READ[REF][i_ind] = y
	
	OUT_READ = head_forward(x, W_READ, B_READ, w_prev, mem)
	
	# read mem
	read_mem = read_from_mem(OUT_READ[O_W], mem)
	
	# output layer
	out = linear_F(out_weights, read_mem.ravel()[:,np.newaxis]) + out_biases
	out += linear_F(out_bypass_weights, x)
	
	return ((out - t)**2).sum()


def g(y):
	#mem[i_ind,j_ind] = y
	
	#out_weights[i_ind,j_ind] = y
	#out_bypass_weights[i_ind,j_ind] = y
	
	W_READ[REF][i_ind,j_ind] = y
	#B_READ[REF][i_ind] = y
	
	OUT_READ = head_forward(x, W_READ, B_READ, w_prev, mem)
	
	# read mem
	read_mem = read_from_mem(OUT_READ[O_W], mem)
	
	# output layer
	out = linear_F(out_weights, read_mem.ravel()[:,np.newaxis]) + out_biases
	out += linear_F(out_bypass_weights, x)
	
	##########
	# backward (data):
	
	# output layer
	dout_dread_mem = linear_dlayer_in(out_weights, 2*(out - t)).reshape((n_controllers, m_length))
	
	# read mem
	dread_from_mem_dw = read_from_mem_dw(mem, dout_dread_mem) # read mem, dw
	dread_from_mem_dmem = read_from_mem_dmem(OUT_READ[O_W], dout_dread_mem) # read mem, dmem ##############################
	
	DW, DB, dmem = head_backward(x, W_READ, B_READ, w_prev, mem, OUT_READ, dread_from_mem_dw)
	
	dout_dout_weights = linear_dF(read_mem.ravel()[:,np.newaxis], 2*(out - t)) # out_weights
	dout_dout_bypass_weights = linear_dF(x, 2*(out - t)) # out_bypass_weights
	
	#return DB[REF][i_ind]
	return DW[REF][i_ind,j_ind]

	#return dmem[i_ind,j_ind] + dread_from_mem_dmem[i_ind,j_ind]
	#return dout_dout_bypass_weights[i_ind,j_ind]
	#return (2*(out - t))[i_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e0


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = W_READ[REF]
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

