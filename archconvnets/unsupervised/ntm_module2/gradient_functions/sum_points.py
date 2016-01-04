import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

def sum_points(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	POINTS = args[0]
	check_buffer(POINTS)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module2.sum_points(POINTS[0], np.prod(POINTS[1]), OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		OUT_BUFFER = set_buffer(return_buffer(POINTS,gpu_ind).sum(), OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (1,)
	return OUT_BUFFER

def sum_points_dinput(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert len(args) == 1
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	POINTS = args[0]
	check_buffer(POINTS)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module2.sum_points_dinput(POINTS[0], np.prod(POINTS[1]), OUT_BUFFER[0], gpu_ind)
	else:
		######### CPU
		temp = np.ones(tuple(np.concatenate(((1,), args[0][1]))),dtype='single')
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = tuple(np.concatenate(((1,), POINTS[1])))
	return OUT_BUFFER
	
def add_sum_layer(LAYERS, name, source=None):
	assert isinstance(name, str)
	assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
	
	# default to previous layer as input
	if source is None:
		in_source = len(LAYERS)-1
		in_shape = LAYERS[in_source]['out_shape']
	# find layer specified
	elif isinstance(source,str):
		in_source = find_layer(LAYERS, source)
		assert in_source is not None, 'could not find source layer %i' % source
		in_shape = LAYERS[in_source]['out_shape']
	else:
		assert False, 'unknown source input'
	
	LAYERS.append({ 'name': name, 'forward_F': sum_points, \
				'out_shape': (1,), \
				'in_shape': [in_shape], \
				'in_source': [in_source], \
				'deriv_F': [sum_points_dinput] })
	
	check_network(LAYERS)
	return LAYERS