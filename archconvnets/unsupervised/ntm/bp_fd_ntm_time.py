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

## head weights
interp_weights = np.random.normal(size=(n_controllers, n_in)) * SCALE
key_weights = np.random.normal(size=(n_controllers*m_length, n_in)) * SCALE

w_prev = np.abs(np.random.normal(size=(n_controllers, n_mem_slots))) * SCALE

mem = np.random.normal(size=(n_mem_slots, m_length)) * SCALE

x = np.random.normal(size=(n_in,1)) * SCALE
x2 = np.random.normal(size=(n_in,1)) * SCALE

t = np.random.normal(size=(n_controllers, m_length)) * SCALE

################# interpolate simplified
def interpolate_simp(w_prev, interp_gate_out):
	# w_prev, w_content: [n_controllers, n_mem_slots], interp_gate_out: [n_controllers, 1]
	return interp_gate_out * w_prev

def interpolate_simp_dinterp_gate_out_int(w_prev, interp_gate_out, dw_prev=0):
	return ( w_prev + interp_gate_out * dw_prev)

def interpolate_simp_dinterp_gate_out(w_prev, interp_gate_out, dw_prev=0, above_w=1):
	return (above_w * ( w_prev + interp_gate_out * dw_prev)).sum(1)[:,np.newaxis] # sum across mem_slots

def interpolate_simp_dw_prev(interp_gate_out, above_w):
	return above_w * ( interp_gate_out)

def f(y):
	#w_prev[i_ind,j_ind] = y
	interp_weights[i_ind,j_ind] = y
	w_prevl = copy.deepcopy(w_prev)
	
	# forward
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate_simp(w_prevl, interp_gate_out)
	read_mem = read_from_mem(w_interp, mem)
	
	dw_prev = interpolate_simp_dinterp_gate_out_int(w_prevl, interp_gate_out)
	#w_prevl = copy.deepcopy(w_interp)
	
	# forward2
	interp_gate_out = linear_F(interp_weights, x2)
	w_interp = interpolate_simp(w_prevl, interp_gate_out)
	read_mem = read_from_mem(w_interp, mem)
	
	return ((read_mem - t)**2).sum()


def g(y):
	#w_prev[i_ind,j_ind] = y
	interp_weights[i_ind,j_ind] = y
	w_prevl = copy.deepcopy(w_prev)
	
	# forward
	interp_gate_out = linear_F(interp_weights, x)
	w_interp = interpolate_simp(w_prevl, interp_gate_out)
	read_mem = read_from_mem(w_interp, mem)
	
	dw_prev = 0*interpolate_simp_dinterp_gate_out_int(w_prevl, interp_gate_out)
	#w_prevl = copy.deepcopy(w_interp)
	
	# forward2
	interp_gate_out = linear_F(interp_weights, x2)
	w_interp = interpolate_simp(w_prevl, interp_gate_out)
	read_mem = read_from_mem(w_interp, mem)
	
	########
	dread_from_mem_dw = read_from_mem_dw(mem, 2*(read_mem - t)) # read mem, dw
	
	# interpolation
	dmem_dw_dw_prev = interpolate_simp_dw_prev(interp_gate_out, dread_from_mem_dw)
	dmem_dw_dw_interp_gate_out = interpolate_simp_dinterp_gate_out(w_prevl, interp_gate_out, dw_prev,\
			dread_from_mem_dw)
	
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

