import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

def softmax(args, OUT_BUFFER=None, gpu_ind=0):
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
		_ntm_module2.softmax(LAYER_IN[0], LAYER_IN[1], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		layer_in = return_buffer(LAYER_IN,gpu_ind)
		exp_layer_in = np.exp(layer_in)
		OUT_BUFFER = set_buffer(exp_layer_in/np.sum(exp_layer_in,1)[:,np.newaxis], OUT_BUFFER, gpu_ind)
	OUT_BUFFER[1] = copy.deepcopy(LAYER_IN[1])
		
	return OUT_BUFFER

def softmax_dlayer_in(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	check_buffer(args[0])
	assert len(args[0][1]) == 2
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	dim1, dim2 = LAYER_OUT[1]
	
	if GPU:
		_ntm_module2.softmax_dlayer_in(LAYER_OUT[0], LAYER_OUT[1], OUT_BUFFER[0], gpu_ind)
	else: 
		############ CPU
		layer_out = return_buffer(LAYER_OUT, gpu_ind)
		
		g = np.zeros((layer_out.shape[0], layer_out.shape[1], layer_out.shape[0], layer_out.shape[1]),dtype='single')
	
		# dsoftmax[:,i]/dlayer_in[:,j] when i = j:
		temp = (layer_out * (1 - layer_out))
		for i in range(g.shape[0]):
			for j in range(g.shape[1]):
				g[i,j,i,j] = temp[i,j]
		
		# i != j
		for i in range(g.shape[0]):
			for j in range(g.shape[1]):
				for k in range(g.shape[1]):
					if j != k:
						g[i,j,i,k] -= layer_out[i,j]*layer_out[i,k]
		
		OUT_BUFFER = set_buffer(g, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = (dim1,dim2,dim1,dim2)
	return OUT_BUFFER

	
def add_softmax_layer(LAYERS, name, source=None, init=0):
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
			in_source = layer_ind-1
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
		
		LAYERS[layer_ind]['forward_F'] = softmax
		LAYERS[layer_ind]['out_shape'] = in_shape[0]
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = [in_source]
		LAYERS[layer_ind]['deriv_F'] = [softmax_dlayer_in]
		LAYERS[layer_ind]['in_prev'] = [False]
		
		return layer_ind

