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
interp_weights = np.random.normal(size=(n_controllers, n_in)) * SCALE
key_weights = np.random.normal(size=(n_controllers*m_length, n_in)) * SCALE

w_prev = np.abs(np.random.normal(size=(n_controllers, n_mem_slots))) * SCALE

mem = np.random.normal(size=(n_mem_slots, m_length)) * SCALE

x = np.random.normal(size=(n_in,1)) * SCALE
x2 = np.random.normal(size=(n_in,1)) * SCALE

t = np.random.normal(size=(n_controllers, m_length)) * SCALE

def f(y):
	interp_weights[i_ind,j_ind] = y
	
	# forward
	keys = (linear_F(key_weights, x)).reshape((n_controllers, m_length))
	w_content = cosine_sim(keys, mem)
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate(w_content, w_prev, interp_gate_out)
	read_mem = read_from_mem(w_interp, mem)
	
	w_new = copy.deepcopy(w_interp)#w_prev)
	
	# forward2
	keys = (linear_F(key_weights, x2)).reshape((n_controllers, m_length))
	w_content = cosine_sim(keys, mem)
	interp_gate_out = linear_F(interp_weights, x2)
	w_interp = interpolate(w_content, w_new, interp_gate_out)
	read_mem = read_from_mem(w_interp, mem)
	
	return ((read_mem - t)**2).sum()


def g(y):
	interp_weights[i_ind,j_ind] = y
	
	# forward
	keys = (linear_F(key_weights, x)).reshape((n_controllers, m_length))
	w_content = cosine_sim(keys, mem)
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate(w_content, w_prev, interp_gate_out)
	read_mem = read_from_mem(w_interp, mem)
	
	w_new = copy.deepcopy(w_interp)#w_prev)
	
	# forward2
	keys = (linear_F(key_weights, x2)).reshape((n_controllers, m_length))
	w_content = cosine_sim(keys, mem)
	interp_gate_out = linear_F(interp_weights, x2)
	w_interp = interpolate(w_content, w_new, interp_gate_out)
	read_mem = read_from_mem(w_interp, mem)
	
	###
	dread_from_mem_dw = read_from_mem_dw(mem, 2*(read_mem - t)) # read mem, dw
	
	# interpolation
	dmem_dw_dw_prev = interpolate_dw_prev(interp_gate_out, dread_from_mem_dw)
	dmem_dw_dw_interp_gate_out = interpolate_dinterp_gate_out(w_content, w_prev, dread_from_mem_dw)
	
	dw_interp_gate_out_dinterp_weights = linear_dF(x2, dmem_dw_dw_interp_gate_out) # interp
	
	return dw_interp_gate_out_dinterp_weights[i_ind,j_ind]
	#return dmem_dw_dw_prev[i_ind,j_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e0


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = interp_weights
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
		
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

