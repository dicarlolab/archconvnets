import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

def sq_points(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	LAYER_IN = args[0]
	check_buffer(LAYER_IN)
	assert len(LAYER_IN[1]) == 2
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		_ntm_module2.sq_points(LAYER_IN[0], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		layer_in = return_buffer(LAYER_IN,gpu_ind)
		OUT_BUFFER = set_buffer(layer_in**2, OUT_BUFFER, gpu_ind)
	OUT_BUFFER[1] = copy.deepcopy(LAYER_IN[1])
		
	return OUT_BUFFER

def sq_points_dinput(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	LAYER_IN = args[0]
	check_buffer(LAYER_IN)
	assert len(LAYER_IN[1]) == 2
	assert LAYER_IN[1] == LAYER_OUT[1]
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	dim1, dim2 = LAYER_OUT[1]
	
	if GPU:
		_ntm_module2.sq_points_dinput(LAYER_IN[0], LAYER_IN[1], OUT_BUFFER[0], gpu_ind)
	else: 
		############ CPU
		input = return_buffer(LAYER_IN, gpu_ind)
		
		n = input.shape[1]
		dinput = np.zeros((input.shape[0], n, input.shape[0], n),dtype='single')
		for i in range(input.shape[0]):
			dinput[i,range(n),i,range(n)] = 2*input[i]
			
		OUT_BUFFER = set_buffer(dinput, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = (dim1,dim2,dim1,dim2)
	return OUT_BUFFER

	
def add_sq_points_layer(LAYERS, name, source=None):
	assert isinstance(name, str)
	assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
	
	# default to previous layer as input
	if source is None:
		in_source = len(LAYERS)-1
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
	
	LAYERS.append({ 'name': name, 'forward_F': sq_points, \
				'out_shape': in_shape[0], \
				'in_shape': in_shape, \
				'in_source': [in_source], \
				'deriv_F': [sq_points_dinput] })
	
	check_network(LAYERS)
	return len(LAYERS)-1

