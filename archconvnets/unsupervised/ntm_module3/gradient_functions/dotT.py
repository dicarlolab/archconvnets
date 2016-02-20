import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0,0]

def dotT(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	_ntm_module3.dotT(F[0], F[1], X[0], X[1], OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = (F[1][0], F[1][-1], X[1][-1])
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

def dotT_da(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	batch_imgs = additional_args[0]
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	_ntm_module3.dotT_da(X[0], F[1], X[1], DERIV_ABOVE[0], OUT_BUFFER[0], gpu_ind)
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	OUT_BUFFER[1] = DERIV_ABOVE[1][:1+n_dim_not_summed] + F[1][1:]
	
	t_main[1] += time.time() - t
	return OUT_BUFFER

def dotT_db(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	batch_imgs = additional_args[0]
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	_ntm_module3.dotT_db(F[0], F[1], X[1], DERIV_ABOVE[0], OUT_BUFFER[0], gpu_ind)
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	OUT_BUFFER[1] = DERIV_ABOVE[1][:1|n_dim_not_summed] + X[1][1:]
	
	t_main[2] += time.time() - t
	return OUT_BUFFER

def add_dotT_layer(LAYERS, name, source, batch_imgs=False, init=0):
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
		in_source = [None]*2
		
		in_source[0] = find_layer(LAYERS, source[0])
		in_source[1] = find_layer(LAYERS, source[1])
		
		assert (in_source[0] is not None) and (in_source[1] is not None)
		
		in_shape[0] = LAYERS[in_source[0]]['out_shape']
		in_shape[1] = LAYERS[in_source[1]]['out_shape']
		
		n_imgs = in_shape[0][0]
		assert n_imgs == in_shape[1][0]
		assert len(in_shape[0]) == len(in_shape[1]) == 3
		
		LAYERS[layer_ind]['forward_F'] = dotT
		LAYERS[layer_ind]['out_shape'] = (n_imgs, in_shape[0][2], in_shape[1][2])
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = in_source
		LAYERS[layer_ind]['deriv_F'] = [dotT_da, dotT_db]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		LAYERS[layer_ind]['additional_forward_args'] = [batch_imgs]
		LAYERS[layer_ind]['additional_deriv_args'] = [[batch_imgs], [batch_imgs]]
		
		return layer_ind
	