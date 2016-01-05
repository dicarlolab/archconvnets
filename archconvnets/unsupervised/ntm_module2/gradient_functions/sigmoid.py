import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

def random_function_1(size):
	return np.asarray(np.random.random(size) +1, dtype='single')

def sigmoid_dlayer_in_test(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	check_buffer(args[0])
	assert len(args[0][1]) == 2
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	dim1, dim2 = LAYER_OUT[1]
	
	_ntm_module2.sigmoid_dlayer_in(LAYER_OUT[0], LAYER_OUT[1], OUT_BUFFER[0], gpu_ind)
	
	############ CPU
	layer_out = return_buffer(LAYER_OUT, gpu_ind)
	d = layer_out * (1-layer_out)
	t = np.zeros(np.concatenate((layer_out.shape, layer_out.shape)),dtype='single')
	for i in range(layer_out.shape[0]):
		for j in range(layer_out.shape[1]):
			t[i,j,i,j] = d[i,j]
	
	OUT_BUFFER[1] = (dim1,dim2,dim1,dim2)
	
	print np.isclose(return_buffer(OUT_BUFFER), t).sum()/np.single(np.prod(OUT_BUFFER[1]))
	print return_buffer(OUT_BUFFER)
	print '..........'
	print t
	return OUT_BUFFER

def sigmoid(args, OUT_BUFFER=None, gpu_ind=0):
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
		_ntm_module2.sigmoid(LAYER_IN[0], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		layer_in = return_buffer(LAYER_IN,gpu_ind)
		OUT_BUFFER = set_buffer(1/(1+np.exp(-layer_in)), OUT_BUFFER, gpu_ind)
	OUT_BUFFER[1] = copy.deepcopy(LAYER_IN[1])
		
	return OUT_BUFFER

def sigmoid_dlayer_in(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
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
		_ntm_module2.sigmoid_dlayer_in(LAYER_OUT[0], LAYER_OUT[1], OUT_BUFFER[0], gpu_ind)
	else: 
		############ CPU
		layer_out = return_buffer(LAYER_OUT, gpu_ind)
		d = layer_out * (1-layer_out)
		t = np.zeros(np.concatenate((layer_out.shape, layer_out.shape)),dtype='single')
		for i in range(layer_out.shape[0]):
			for j in range(layer_out.shape[1]):
				t[i,j,i,j] = d[i,j]
		OUT_BUFFER = set_buffer(t, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = (dim1,dim2,dim1,dim2)
	return OUT_BUFFER


def add_sigmoid_layer(LAYERS, name, source=None):
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
	
	LAYERS.append({ 'name': name, 'forward_F': sigmoid, \
				'out_shape': in_shape[0], \
				'in_shape': in_shape, \
				'in_source': [in_source], \
				'deriv_F': [sigmoid_dlayer_in] })
	
	check_network(LAYERS)
	return len(LAYERS)-1