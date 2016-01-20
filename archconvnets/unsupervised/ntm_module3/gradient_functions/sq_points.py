import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0]

# deriv_computable: require dimensions be <= 2, required for sq_points_dinput to work correctly
def sq_points(args, OUT_BUFFER=None, additional_args=[None], deriv_computable=True, gpu_ind=0):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	assert len(args) == 1
	LAYER_IN = args[0]
	check_buffer(LAYER_IN)
	if deriv_computable:
		assert len(LAYER_IN[1]) <= 2
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		_ntm_module3.sq_points(LAYER_IN[0], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		layer_in = return_buffer(LAYER_IN,gpu_ind)
		OUT_BUFFER = set_buffer(layer_in**2, OUT_BUFFER, gpu_ind)
	OUT_BUFFER[1] = copy.deepcopy(LAYER_IN[1])
	t_main[0] += time.time() - t
	return OUT_BUFFER

def sq_points_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=0):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	assert len(args) == 1
	LAYER_IN = args[0]
	check_buffer(LAYER_IN)
	check_buffer(DERIV_ABOVE)
	assert len(LAYER_IN[1]) <= 2
	assert LAYER_IN[1] == LAYER_OUT[1]
	
	LAYER_IN_R = copy.deepcopy(LAYER_IN)
	if len(LAYER_IN[1]) == 1:
		LAYER_IN_R[1] = (LAYER_IN_R[1][0],1)
		
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	dim1, dim2 = LAYER_IN_R[1]
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.sq_points_dinput(LAYER_IN[0], LAYER_IN_R[1], OUT_BUFFER_TEMP[0], gpu_ind)
	else: 
		############ CPU
		input = return_buffer(LAYER_IN_R, gpu_ind)
		
		n = input.shape[1]
		dinput = np.zeros((input.shape[0], n, input.shape[0], n),dtype='single')
		for i in range(input.shape[0]):
			dinput[i,range(n),i,range(n)] = 2*input[i]
			
		OUT_BUFFER_TEMP = set_buffer(dinput, OUT_BUFFER_TEMP, gpu_ind)
	
	if len(LAYER_IN[1]) == 1:
		OUT_BUFFER_TEMP[1] = (dim1,dim1)
	else:
		OUT_BUFFER_TEMP[1] = (dim1,dim2,dim1,dim2)
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	t_main[1] += time.time() - t
	return OUT_BUFFER

	
def add_sq_points_layer(LAYERS, name, source=None, init=0):
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
			in_source = layer_ind - 1
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
		
		LAYERS[layer_ind]['forward_F'] = sq_points
		LAYERS[layer_ind]['out_shape'] = in_shape[0]
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = [in_source]
		LAYERS[layer_ind]['deriv_F'] = [sq_points_dinput]
		LAYERS[layer_ind]['in_prev'] = [False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None]]
		
		return layer_ind

