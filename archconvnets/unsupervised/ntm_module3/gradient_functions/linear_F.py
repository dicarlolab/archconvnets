import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *

def random_function(size):
	return np.asarray(np.random.random(size) - .5, dtype='single')

# additional_args = [True]: squeeze output last dimension
def linear_F_dx(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[True], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	check_buffer(DERIV_ABOVE)
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
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.linear_F_dx(F_new_shape[0], X_reshaped[1], F_new_shape[1], OUT_BUFFER_TEMP[0], gpu_ind)
	else: 
		############ CPU
		f = return_buffer(F_new_shape, gpu_ind)
		x = return_buffer(X_reshaped, gpu_ind)
		n = x.shape[1]
		temp = np.zeros((f.shape[0], n, x.shape[0], n),dtype='single')
		temp[:,range(n),:,range(n)] = f
		OUT_BUFFER_TEMP = set_buffer(temp, OUT_BUFFER_TEMP, gpu_ind)
	
	OUT_BUFFER_TEMP[1] = tuple(np.concatenate((LAYER_OUT[1], np.asarray(X[1]))))
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	
	return OUT_BUFFER

# additional_args = [True]: squeeze output last dimension
def linear_F_dF(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[True], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert GPU
	F, X = args
	check_buffer(F)
	check_buffer(X)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	check_buffer(DERIV_ABOVE)
	assert len(F[1]) >= 2
	assert len(X[1]) == 2 or len(X[1]) == 4
	
	# if source is a conv layer (4D input), sum across everything
	X_reshaped = copy.deepcopy(X)
	if len(X[1]) == 4:
		X_reshaped[1] = (np.prod(X[1]), 1)
	
	assert F[1][-1] == X_reshaped[1][0]
	
	X_dim0, X_dim1 = X_reshaped[1]
	
	# reshape deriv_above to 2 dims
	DERIV_ABOVE_reshaped = copy.deepcopy(DERIV_ABOVE)
	DERIV_ABOVE_reshaped[1] = (np.prod(DERIV_ABOVE[1][:len(DERIV_ABOVE[1])-1]), DERIV_ABOVE[1][-1])
	
	# now: dot(deriv_above, x.T)
	_ntm_module3.linear_F_dF(X_reshaped[0], X_reshaped[1], DERIV_ABOVE_reshaped[0], DERIV_ABOVE_reshaped[1], OUT_BUFFER[0], gpu_ind)
	
	# reshape back to original dimensions
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dim_not_summed], F[1])))
	check_buffer(OUT_BUFFER)
	
	return OUT_BUFFER

linear_F = dot

def add_linear_F_layer(LAYERS, name, n_filters, source=None, squeeze=True, random_function=random_function, init=0):
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
		