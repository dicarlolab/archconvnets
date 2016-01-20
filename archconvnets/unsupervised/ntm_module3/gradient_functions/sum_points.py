import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0]

def sum_points(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=0):
	t = time.time()
	assert additional_args == [None]
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	POINTS = args[0]
	check_buffer(POINTS)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module3.sum_points(POINTS[0], np.prod(POINTS[1]), OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		OUT_BUFFER = set_buffer(return_buffer(POINTS,gpu_ind).sum(), OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (1,)
	t_main[0] += time.time() - t
	return OUT_BUFFER

def sum_points_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=0):
	t = time.time()
	assert additional_args == [None]
	assert len(args) == 1
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	POINTS = args[0]
	check_buffer(POINTS)
	check_buffer(LAYER_OUT)
	check_buffer(DERIV_ABOVE)
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	points = return_buffer(POINTS)
	deriv_above = return_buffer(DERIV_ABOVE)
	
	DERIV_ABOVE_shape = DERIV_ABOVE[1]
	POINTS_shape = POINTS[1]
	
	# DERIV_ABOVE: (a,b,c, 1)
	# LAYER_OUT: (1)
	# n_dims_not_summed: a,b,c
	# POINTS: (g,h,i,j)	
	
	# exclude/sume over layer_output dimension:
	dims_keep = np.concatenate((range(deriv_above.ndim - 1),  deriv_above.ndim + np.arange(points.ndim)))
	
	output = np.einsum(deriv_above, range(deriv_above.ndim), np.ones_like(points), deriv_above.ndim + np.arange(points.ndim), dims_keep)
	
	OUT_BUFFER = set_buffer(output, OUT_BUFFER)
	t_main[1] += time.time() - t
	return OUT_BUFFER
	
def add_sum_layer(LAYERS, name, source=None, init=0):
	t = time.time()
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
			in_shape = LAYERS[in_source]['out_shape']
		# find layer specified
		elif isinstance(source,str):
			in_source = find_layer(LAYERS, source)
			assert in_source is not None, 'could not find source layer %i' % source
			in_shape = LAYERS[in_source]['out_shape']
		else:
			assert False, 'unknown source input'
		
		LAYERS[layer_ind]['forward_F'] = sum_points
		LAYERS[layer_ind]['out_shape'] = (1,)
		LAYERS[layer_ind]['in_shape'] = [in_shape]
		LAYERS[layer_ind]['in_source'] = [in_source]
		LAYERS[layer_ind]['deriv_F'] = [sum_points_dinput]
		LAYERS[layer_ind]['in_prev'] = [False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None]]
		
		return layer_ind
