import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0]

def random_function_1(size):
	return np.asarray(np.random.random(size) +1, dtype='single')

def sigmoid(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	LAYER_IN = args[0]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	_ntm_module3.sigmoid(LAYER_IN[0], OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = LAYER_IN[1][:2]
	
	if DEBUG:
		assert additional_args == [None]
		assert isinstance(gpu_ind,int)
		assert len(args) == 1
		check_buffer(LAYER_IN)
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

def sigmoid_dlayer_in(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	LAYER_IN = args[0]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	# reshape deriv_above to 2 dims
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	DERIV_ABOVE_reshaped = (np.prod(DERIV_ABOVE[1][:n_dim_not_summed]), np.prod(DERIV_ABOVE[1][n_dim_not_summed:]))
	
	# deriv_above * layer_out * (1-layer_out)
	_ntm_module3.sigmoid_dlayer_in(LAYER_OUT[0], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dim_not_summed], LAYER_IN[1])))
	
	if DEBUG:
		assert additional_args == [None]
		assert isinstance(gpu_ind,int)
		assert len(args) == 1
		assert GPU
		assert LAYER_IN[1][:2] == LAYER_OUT[1]
		check_buffer(LAYER_OUT)
		check_buffer(args[0])
		check_buffer(OUT_BUFFER)
		check_buffer(OUT_BUFFER)
		check_buffer(DERIV_ABOVE)
		
	t_main[1] += time.time() - t
	return OUT_BUFFER


def add_sigmoid_layer(LAYERS, name, source=None, init=0):
	assert isinstance(name, str)
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		# default to previous layer as input
		if source is None:
			in_source = layer_ind-1
			in_shape = [LAYERS[in_source]['out_shape']]
		# find layer specified
		elif isinstance(source,str):
			in_source = find_layer(LAYERS, source)
			assert in_source is not None, 'could not find source layer %i' % source
			in_shape = [LAYERS[in_source]['out_shape']]
		
		# input is user supplied
		elif isinstance(source,tuple):
			in_shape = [source]
			in_source = -1
		else:
			assert False, 'unknown source input'
		
		LAYERS[layer_ind]['forward_F'] = sigmoid
		LAYERS[layer_ind]['out_shape'] = in_shape[0][:2]
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = [in_source]
		LAYERS[layer_ind]['deriv_F'] = [sigmoid_dlayer_in]
		LAYERS[layer_ind]['in_prev'] = [False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None]]
		
		return layer_ind
