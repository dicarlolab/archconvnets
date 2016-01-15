import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

def random_function(size):
	return np.asarray(np.random.random(size) - .5, dtype='single')

def linear_F_dx(args, LAYER_OUT, OUT_BUFFER=None, additional_args=[True], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) >= 2
	assert len(X[1]) == 2 or len(X[1]) == 4
	
	# if source is a conv layer (4D input), sum across everything
	X_reshaped = copy.deepcopy(X)
	if len(X[1]) == 4:
		X_reshaped[1] = (np.prod(X[1]), 1)
	
	assert F[1][-1] == X_reshaped[1][0]
	
	# reshape buffer1 into two dimensions:
	# (a,b,c,d,e) -> (a*b*c*d, e)
	F_new_shape = copy.deepcopy(F)
	F_new_shape[1] = (np.prod(F[1][:len(F[1])-1]), F[1][-1])
	
	F_dim0, F_dim1 = F_new_shape[1]
	X_dim0, X_dim1 = X_reshaped[1]
	
	if GPU:
		_ntm_module2.linear_F_dx(F_new_shape[0], X_reshaped[1], F_new_shape[1], OUT_BUFFER[0], gpu_ind)
	else: 
		############ CPU
		f = return_buffer(F_new_shape, gpu_ind)
		x = return_buffer(X_reshaped, gpu_ind)
		n = x.shape[1]
		temp = np.zeros((f.shape[0], n, x.shape[0], n),dtype='single')
		temp[:,range(n),:,range(n)] = f
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
	
	#### forward out shape:
	forward_shape = tuple(np.concatenate((np.asarray(F_new_shape[1][:len(F_new_shape[1])-1]), np.asarray(X_reshaped[1][1])[np.newaxis])))	
	if additional_args[0] and forward_shape[-1] == 1: # squeeze
		forward_shape = forward_shape[:len(forward_shape)-1]
	###
	
	OUT_BUFFER[1] = tuple(np.concatenate((forward_shape, np.asarray(X[1]))))
	
	return OUT_BUFFER

def linear_F_dF(args, LAYER_OUT, OUT_BUFFER=None, additional_args=[True], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) >= 2
	assert len(X[1]) == 2 or len(X[1]) == 4
	
	# if source is a conv layer (4D input), sum across everything
	X_reshaped = copy.deepcopy(X)
	if len(X[1]) == 4:
		X_reshaped[1] = (np.prod(X[1]), 1)
	
	assert F[1][-1] == X_reshaped[1][0]
	
	# reshape buffer1 into two dimensions:
	# (a,b,c,d,e) -> (a*b*c*d, e)
	F_new_shape = copy.deepcopy(F)
	F_new_shape[1] = (np.prod(F[1][:len(F[1])-1]), F[1][-1])
	
	F_dim0, F_dim1 = F_new_shape[1]
	X_dim0, X_dim1 = X_reshaped[1]
	
	if GPU:
		_ntm_module2.linear_F_dF(X_reshaped[0], X_reshaped[1], F_new_shape[1], OUT_BUFFER[0], gpu_ind)
	else:
		############ CPU
		f = return_buffer(F_new_shape, gpu_ind)
		x = return_buffer(X_reshaped, gpu_ind)
		n = f.shape[0]
		temp = np.zeros((n, x.shape[1], n, f.shape[1]),dtype='single')
		temp[range(n),:,range(n)] = x.T
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
		
	#### forward out shape:
	forward_shape = tuple(np.concatenate((np.asarray(F_new_shape[1][:len(F_new_shape[1])-1]), np.asarray(X_reshaped[1][1])[np.newaxis])))	
	if additional_args[0] and forward_shape[-1] == 1: # squeeze
		forward_shape = forward_shape[:len(forward_shape)-1]
	###
	
	OUT_BUFFER[1] = tuple(np.concatenate((forward_shape, np.asarray(F[1]))))
	
	return OUT_BUFFER
	
linear_F = dot

def add_linear_F_layer(LAYERS, name, n_filters, source=None, squeeze=False, random_function=random_function, init=0):
	assert isinstance(name, str)
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		in_shape = [None]*2
		in_prev1 = False
		
		# default to previous layer as input
		if source is None:
			in_source = layer_ind-1
			in_shape[1] = LAYERS[in_source]['out_shape']
		# find layer specified
		elif isinstance(source,str):
			in_source = find_layer(LAYERS, source)
			assert in_source is not None, 'could not find source layer %i' % source
			in_shape[1] = LAYERS[in_source]['out_shape']
			in_prev1 = source[-1] == '-'
		
		# input is user supplied
		elif isinstance(source,tuple):
			in_shape[1] = source
			in_source = -1
		else:
			assert False, 'unknown source input'
		
		# if source is a conv layer (4D input), sum across everything
		assert len(in_shape[1]) == 4 or len(in_shape[1]) == 2
		if len(in_shape[1]) == 4:
			in_shape_reshaped = (np.prod(in_shape[1]), 1)
		else:
			in_shape_reshaped = copy.deepcopy(in_shape[1])
		
		# if n_filters is an int or a tuple
		if isinstance(n_filters,int):
			in_shape[0] = (n_filters, in_shape_reshaped[0])
			out_shape = (in_shape[0][0], in_shape_reshaped[1])
		else:
			in_shape[0] = tuple(np.concatenate((np.asarray(n_filters), np.asarray(in_shape_reshaped[0])[np.newaxis])))
			out_shape = tuple(np.concatenate((in_shape[0][:len(in_shape[0])-1], np.asarray(in_shape_reshaped[1])[np.newaxis])))
		
		if squeeze and out_shape[-1] == 1:
			out_shape = out_shape[:len(out_shape)-1]
		
		LAYERS[layer_ind]['forward_F'] = linear_F
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = [random_function, in_source]
		LAYERS[layer_ind]['deriv_F'] = [linear_F_dF, linear_F_dx]
		LAYERS[layer_ind]['in_prev'] = [False, in_prev1]
		LAYERS[layer_ind]['additional_forward_args'] = [squeeze]
		LAYERS[layer_ind]['additional_deriv_args'] = [[squeeze], [squeeze]]
		
		return layer_ind
		