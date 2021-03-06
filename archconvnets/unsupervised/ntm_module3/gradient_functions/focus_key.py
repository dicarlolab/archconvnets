import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0,0]

##########
# focus keys, scalar beta_out (one for each controller) multiplied with each of its keys
def focus_keys(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	batch_imgs = additional_args[0]
	KEYS, BETA_OUT = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	_ntm_module3.focus_key(KEYS[0], KEYS[1], BETA_OUT[0], OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = KEYS[1]
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

def focus_key_dbeta_out(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	batch_imgs = additional_args[0]
	KEYS, BETA_OUT = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	_ntm_module3.focus_key_dbeta_out(KEYS[0], KEYS[1], DERIV_ABOVE[0], OUT_BUFFER[0], gpu_ind)
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed+1] + BETA_OUT[1][1:]
	
	t_main[1] += time.time() - t
	return OUT_BUFFER

def focus_key_dkeys(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	batch_imgs = additional_args[0]
	KEYS, BETA_OUT = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	_ntm_module3.focus_key_dkeys(BETA_OUT[0], KEYS[1], DERIV_ABOVE[0], OUT_BUFFER[0], gpu_ind)
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed+1] + KEYS[1][1:]
	
	t_main[2] += time.time() - t
	return OUT_BUFFER

def add_focus_keys_layer(LAYERS, name, source, init=0):
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
		
		source[1] = find_layer(LAYERS, source[1])
		assert source[1] is not None, 'could not find source layer 1'
		
		in_shape[0] = LAYERS[source[0]]['out_shape']
		in_shape[1] = in_shape[0][:2] + (1,)
		
		LAYERS[layer_ind]['forward_F'] = focus_keys
		LAYERS[layer_ind]['out_shape'] = LAYERS[source[0]]['out_shape']
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = source
		LAYERS[layer_ind]['deriv_F'] = [focus_key_dkeys, focus_key_dbeta_out]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None], [None]]
		
		return layer_ind
