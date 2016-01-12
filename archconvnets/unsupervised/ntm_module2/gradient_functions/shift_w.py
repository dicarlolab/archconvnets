import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

##########
N_SHIFTS = 3

def shift_w_test(args, OUT_BUFFER=None, gpu_ind=0):
	# shift_out: [n_controllers, n_shifts], w_interp: [n_controllers, mem_length]
	assert isinstance(gpu_ind,int)
	SHIFT_OUT, W_INTERP = args
	check_buffer(SHIFT_OUT)
	check_buffer(W_INTERP)
	assert SHIFT_OUT[1][0] == W_INTERP[1][0]
	assert len(SHIFT_OUT[1]) == len(W_INTERP[1]) == 2
	assert SHIFT_OUT[1][1] == 3 # 3 shifts
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	_ntm_module2.shift_w(SHIFT_OUT[0], W_INTERP[0], W_INTERP[1], OUT_BUFFER[0], gpu_ind)
	
	######## CPU
	shift_out = return_buffer(SHIFT_OUT,gpu_ind)
	w_interp = return_buffer(W_INTERP,gpu_ind)
	
	w_tilde = np.zeros_like(w_interp)
	n_mem_slots = w_interp.shape[1]
	
	for loc in range(n_mem_slots):
		w_tilde[:,loc] = shift_out[:,0]*w_interp[:,loc-1] + shift_out[:,1]*w_interp[:,loc] + \
				shift_out[:,2]*w_interp[:,(loc+1)%n_mem_slots]
		
	OUT_BUFFER[1] = copy.deepcopy(W_INTERP[1])
	z = return_buffer(OUT_BUFFER)
	
	print z
	print '...'
	print w_tilde
	print np.isclose(z, w_tilde).sum()/np.single(np.prod(z.shape))
	
	return OUT_BUFFER

def shift_w(args, OUT_BUFFER=None, gpu_ind=0):
	# shift_out: [n_controllers, n_shifts], w_interp: [n_controllers, mem_length]
	assert isinstance(gpu_ind,int)
	SHIFT_OUT, W_INTERP = args
	check_buffer(SHIFT_OUT)
	check_buffer(W_INTERP)
	assert SHIFT_OUT[1][0] == W_INTERP[1][0]
	assert len(SHIFT_OUT[1]) == len(W_INTERP[1]) == 2
	assert SHIFT_OUT[1][1] == 3 # 3 shifts
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module2.shift_w(SHIFT_OUT[0], W_INTERP[0], W_INTERP[1], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		shift_out = return_buffer(SHIFT_OUT,gpu_ind)
		w_interp = return_buffer(W_INTERP,gpu_ind)
		
		w_tilde = np.zeros_like(w_interp)
		n_mem_slots = w_interp.shape[1]
		
		for loc in range(n_mem_slots):
			w_tilde[:,loc] = shift_out[:,0]*w_interp[:,loc-1] + shift_out[:,1]*w_interp[:,loc] + \
					shift_out[:,2]*w_interp[:,(loc+1)%n_mem_slots]
		OUT_BUFFER = set_buffer(w_tilde, OUT_BUFFER, gpu_ind) # [n_controllers, m_length]
		
	OUT_BUFFER[1] = copy.deepcopy(W_INTERP[1])
	return OUT_BUFFER

def shift_w_dshift_out(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	SHIFT_OUT, W_INTERP = args
	check_buffer(SHIFT_OUT)
	check_buffer(W_INTERP)
	assert SHIFT_OUT[1][0] == W_INTERP[1][0]
	assert len(SHIFT_OUT[1]) == len(W_INTERP[1]) == 2
	assert SHIFT_OUT[1][1] == 3 # 3 shifts
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	C, M = W_INTERP[1]
	n_shifts = 3 #...
	
	if GPU:
		_ntm_module2.shift_w_dshift_out(W_INTERP[0], W_INTERP[1], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		w_interp = return_buffer(W_INTERP, gpu_ind)
	
		temp = np.zeros((C, M, C, n_shifts),dtype='single')
		for m in range(M):
			for H in [-1,0,1]:
				temp[range(C),m,range(C),H+1] = w_interp[:, (m+H)%M]
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (C, M, C, n_shifts)
	return OUT_BUFFER

def shift_w_dw_interp(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	SHIFT_OUT, W_INTERP = args
	check_buffer(SHIFT_OUT)
	check_buffer(W_INTERP)
	assert SHIFT_OUT[1][0] == W_INTERP[1][0]
	assert len(SHIFT_OUT[1]) == len(W_INTERP[1]) == 2
	assert SHIFT_OUT[1][1] == 3 # 3 shifts
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	C, M = W_INTERP[1]
	n_shifts = 3 #...
	
	if GPU:
		_ntm_module2.shift_w_dw_interp(SHIFT_OUT[0], W_INTERP[1], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		shift_out = return_buffer(SHIFT_OUT, gpu_ind)
	
		temp = np.zeros((C, M, C, M),dtype='single')
	
		for loc in range(M):
			temp[range(C),loc,range(C),loc-1] = shift_out[:,0]
			temp[range(C),loc,range(C),loc] = shift_out[:,1]
			temp[range(C),loc,range(C),(loc+1)%M] = shift_out[:,2]
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (C, M, C, M)
	return OUT_BUFFER

def add_shift_w_layer(LAYERS, name, source, init=0):
	assert isinstance(name, str)
	assert len(source) == 2
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		in_shape = [None]*2
		
		source[0] = find_layer(LAYERS, source[0])
		assert source[0] is not None, 'could not find source layer 0'
		
		assert isinstance(source[1],str)
		source[1] = find_layer(LAYERS, source[1])
		
		
		in_shape[1] = LAYERS[source[1]]['out_shape']
		in_shape[0] = (in_shape[1][0], N_SHIFTS)
		
		LAYERS[layer_ind]['forward_F'] = shift_w
		LAYERS[layer_ind]['out_shape'] = LAYERS[source[1]]['out_shape']
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = source
		LAYERS[layer_ind]['deriv_F'] = [shift_w_dshift_out, shift_w_dw_interp]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		
		return layer_ind