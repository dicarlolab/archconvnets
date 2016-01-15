import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *

def max_pool(args, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	CONV_OUTPUT = args[0]
	check_buffer(CONV_OUTPUT)
	assert len(CONV_OUTPUT[1]) == 4
	assert CONV_OUTPUT[1][-1] == CONV_OUTPUT[1][-2]
	assert CONV_OUTPUT[1][0] == 1
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		OUT_BUFFER[1] = _ntm_module3.max_pool(CONV_OUTPUT[0], CONV_OUTPUT[1], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		assert False, 'cpu max_pool not supported'
		
	return OUT_BUFFER

# srcData = LAYER_OUT
# destData = args[0]
def max_pool_dinput(args, LAYER_OUT, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	DESTDATA = args[0]
	SRCDATA = LAYER_OUT
	
	check_buffer(SRCDATA)
	check_buffer(DESTDATA)
	
	assert len(SRCDATA[1]) == len(DESTDATA[1]) == 4
	
	assert SRCDATA[1][-1] == SRCDATA[1][-2]
	assert DESTDATA[1][-1] == DESTDATA[1][-2]

	assert SRCDATA[1][0] == DESTDATA[1][0]
	assert SRCDATA[1][1] == DESTDATA[1][1]
	
	assert SRCDATA[1][0] == 1
	
	######## use identity matrix so we get derivs wrt to each output location
	layer_out_shape = SRCDATA[1]
	
	deriv_above = np.zeros((layer_out_shape[0], np.prod(layer_out_shape[1:]), np.prod(layer_out_shape[1:])), dtype='single')
	deriv_above[range(layer_out_shape[0])] = np.eye(np.prod(layer_out_shape[1:]))
	deriv_above = deriv_above.reshape(np.concatenate((np.prod(layer_out_shape)[np.newaxis], layer_out_shape[1:])))
	
	DERIV_ABOVE = init_buffer(deriv_above, gpu_ind=gpu_ind)
	#######
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		OUT_BUFFER[1] = _ntm_module3.max_pool_dinput(DESTDATA[0], DESTDATA[1], SRCDATA[0], DERIV_ABOVE[0], DERIV_ABOVE[1], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		assert False, 'cpu max_pool not supported'
	
	free_buffer(DERIV_ABOVE)
	
	OUT_BUFFER[1] = tuple(np.concatenate((LAYER_OUT[1], DESTDATA[1])))
	
	return OUT_BUFFER


# source = None: source is previous layer
# source = -1: source is user-supplied
# source = str: source is another layer
def add_max_pool_layer(LAYERS, name, init=0):
	assert isinstance(name, str)
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		assert layer_ind >= 0
		
		in_shape = LAYERS[layer_ind-1]['out_shape']
		
		# empirically determine output shape
		IMGS_temp = init_buffer(np.zeros(in_shape, dtype='single'))
		
		O = max_pool((IMGS_temp,))
		out_shape = copy.deepcopy(O[1])
		
		free_buffer(O)
		free_buffer(IMGS_temp)
		
		LAYERS[layer_ind]['forward_F'] = max_pool
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = [in_shape]
		LAYERS[layer_ind]['in_source'] = [layer_ind-1]
		LAYERS[layer_ind]['deriv_F'] = [max_pool_dinput]
		LAYERS[layer_ind]['in_prev'] = [False]
		
		return layer_ind
