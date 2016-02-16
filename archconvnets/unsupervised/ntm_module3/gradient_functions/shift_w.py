import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0,0]

##########
N_SHIFTS = 3

def shift_w(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	# shift_out: [n_controllers, n_shifts], w_interp: [n_controllers, mem_length]
	
	SHIFT_OUT, W_INTERP = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	
	_ntm_module3.shift_w(SHIFT_OUT[0], W_INTERP[0], W_INTERP[1], OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = W_INTERP[1]
	
	if DEBUG:
		assert isinstance(gpu_ind,int)
		assert additional_args == [None]
		check_buffer(OUT_BUFFER)
		check_buffer(SHIFT_OUT)
		check_buffer(W_INTERP)
		assert SHIFT_OUT[1][0] == W_INTERP[1][0]
		assert len(SHIFT_OUT[1]) == len(W_INTERP[1]) == 2
		assert SHIFT_OUT[1][1] == 3 # 3 shifts
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

def shift_w_dshift_out(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	SHIFT_OUT, W_INTERP = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	C, M = W_INTERP[1]
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	DERIV_ABOVE_reshaped = tuple(np.concatenate((np.prod(DERIV_ABOVE[1][:n_dim_not_summed])[np.newaxis], DERIV_ABOVE[1][n_dim_not_summed:])))
	
	_ntm_module3.shift_w_dshift_out(W_INTERP[0], W_INTERP[1], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dim_not_summed], SHIFT_OUT[1])))
	
	if DEBUG:
		assert isinstance(gpu_ind,int)
		assert additional_args == [None]
		check_buffer(SHIFT_OUT)
		check_buffer(W_INTERP)
		assert SHIFT_OUT[1][0] == W_INTERP[1][0]
		assert len(SHIFT_OUT[1]) == len(W_INTERP[1]) == 2
		assert SHIFT_OUT[1][1] == 3 # 3 shifts
		check_buffer(OUT_BUFFER)
		check_buffer(DERIV_ABOVE)
		check_buffer(OUT_BUFFER_TEMP)
	
	t_main[1] += time.time() - t
	return OUT_BUFFER

def shift_w_dw_interp(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	SHIFT_OUT, W_INTERP = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	DERIV_ABOVE_reshaped = tuple(np.concatenate((np.prod(DERIV_ABOVE[1][:n_dim_not_summed])[np.newaxis], DERIV_ABOVE[1][n_dim_not_summed:])))
	
	_ntm_module3.shift_w_dw_interp(SHIFT_OUT[0], W_INTERP[1], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dim_not_summed], W_INTERP[1])))
	
	if DEBUG:
		assert isinstance(gpu_ind,int)
		assert additional_args == [None]
		check_buffer(SHIFT_OUT)
		check_buffer(W_INTERP)
		assert SHIFT_OUT[1][0] == W_INTERP[1][0]
		assert len(SHIFT_OUT[1]) == len(W_INTERP[1]) == 2
		assert SHIFT_OUT[1][1] == 3 # 3 shifts
		check_buffer(OUT_BUFFER)
		check_buffer(DERIV_ABOVE)
	
	t_main[2] += time.time() - t
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
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None], [None]]
		
		return layer_ind
