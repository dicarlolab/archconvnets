import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *

def dotT(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	BUFFER1, BUFFER2 = args
	check_buffer(BUFFER1)
	check_buffer(BUFFER2)
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	check_buffer(OUT_BUFFER)
	assert len(BUFFER1[1]) == len(BUFFER2[1]) == 2
	assert BUFFER1[1][0] == BUFFER2[1][0]
	assert OUT_BUFFER[0] != BUFFER1[0]
	assert OUT_BUFFER[0] != BUFFER2[0]
	
	if GPU:
		_ntm_module3.dotT(BUFFER1[0], BUFFER1[1], BUFFER2[0], BUFFER2[1], OUT_BUFFER[0], gpu_ind)
	else:
		######### CPU
		F = return_buffer(BUFFER1, gpu_ind)
		x = return_buffer(BUFFER2, gpu_ind)
		temp = np.asarray(np.dot(F.T,x),dtype='single') # [n1, 1]
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (BUFFER1[1][1], BUFFER2[1][1])
	return OUT_BUFFER

def dotT_da(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) == len(X[1]) == 2
	assert F[1][0] == X[1][0]
	
	F_dim0, F_dim1 = F[1]
	X_dim0, X_dim1 = X[1]
	
	if GPU:
		_ntm_module3.dotT_da(X[0], F[1], X[1], OUT_BUFFER[0], gpu_ind)
	else: 
		############ CPU
		x = return_buffer(X, gpu_ind)
		
		temp = np.zeros((F_dim1, X_dim1, F_dim0, F_dim1),dtype='single')
		temp[range(F_dim1),:,:,range(F_dim1)] = x.T
		
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = (F_dim1, X_dim1, F_dim0, F_dim1)
	return OUT_BUFFER

def dotT_db(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) == len(X[1]) == 2
	assert F[1][0] == X[1][0]
	
	F_dim0, F_dim1 = F[1]
	X_dim0, X_dim1 = X[1]
	
	if GPU:
		_ntm_module3.dotT_db(F[0], F[1], X[1], OUT_BUFFER[0], gpu_ind)
	else: 
		############ CPU
		f = return_buffer(F, gpu_ind)
		
		temp = np.zeros((F_dim1, X_dim1, X_dim0, X_dim1),dtype='single')
		temp[:,range(X_dim1),:,range(X_dim1)] = f.T
		
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = (F_dim1, X_dim1, X_dim0, X_dim1)
	return OUT_BUFFER

def add_dotT_layer(LAYERS, name, source, init=0):
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
		
		LAYERS[layer_ind]['forward_F'] = dotT
		LAYERS[layer_ind]['out_shape'] = (in_shape[0][1], in_shape[1][1])
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = in_source
		LAYERS[layer_ind]['deriv_F'] = [dotT_da, dotT_db]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		
		return layer_ind
	