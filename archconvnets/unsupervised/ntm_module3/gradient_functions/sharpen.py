import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0,0]

############# sharpen across mem_slots separately for each controller
# w: [dim1, dim0]
# gamma: [dim1, 1]
def sharpen(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	W, GAMMA = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	_ntm_module3.sharpen(W[0], W[1], GAMMA[0], OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = W[1]
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

def sharpen_dgamma(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	W, GAMMA = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	_ntm_module3.sharpen_dgamma(W[0], W[1], GAMMA[0], DERIV_ABOVE[0], OUT_BUFFER[0], gpu_ind)
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed+1] + GAMMA[1][1:]
	
	t_main[1] += time.time() - t
	return OUT_BUFFER
	
def sharpen_dw(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	W, GAMMA = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	_ntm_module3.sharpen_dw(W[0], W[1], GAMMA[0], DERIV_ABOVE[0], OUT_BUFFER[0], gpu_ind)
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed+1] + W[1][1:]
	
	t_main[2] += time.time() - t
	return OUT_BUFFER

def add_sharpen_layer(LAYERS, name, source, init=0):
	assert isinstance(name, str)
	assert isinstance(source, list)
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
		
		if isinstance(source[1],int) != True and source[1] != -1:
			source[1] = find_layer(LAYERS, source[1])
		
		in_shape[0] = LAYERS[source[0]]['out_shape']
		in_shape[1] = in_shape[0][:2] + (1,)
		
		assert len(in_shape[0]) == 3
		
		LAYERS[layer_ind]['forward_F'] = sharpen
		LAYERS[layer_ind]['out_shape'] = LAYERS[source[0]]['out_shape']
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = source
		LAYERS[layer_ind]['deriv_F'] = [sharpen_dw, sharpen_dgamma]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None], [None]]
		
		return layer_ind
