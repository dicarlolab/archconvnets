import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0,0]

def softmax(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	assert len(args) == 1
	LAYER_IN = args[0]
	check_buffer(LAYER_IN)
	assert (len(LAYER_IN[1]) == 2) or ((len(LAYER_IN[1]) == 3) and (LAYER_IN[1][2] == 1))
	
	LAYER_IN_R = copy.deepcopy(LAYER_IN)
	LAYER_IN_R[1] = LAYER_IN_R[1][:2]
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		_ntm_module3.softmax(LAYER_IN_R[0], LAYER_IN_R[1], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		layer_in = return_buffer(LAYER_IN_R,gpu_ind)
		exp_layer_in = np.exp(layer_in)
		OUT_BUFFER = set_buffer(exp_layer_in/np.sum(exp_layer_in,1)[:,np.newaxis], OUT_BUFFER, gpu_ind)
	OUT_BUFFER[1] = copy.deepcopy(LAYER_IN_R[1])
	t_main[0] += time.time() - t
	return OUT_BUFFER

def softmax_dlayer_in(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	assert len(args) == 1
	check_buffer(args[0])
	check_buffer(DERIV_ABOVE)
	assert (len(args[0][1]) == 2) or ((len(args[0][1]) == 3) and (args[0][1][2] == 1))
	
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	dim1, dim2 = LAYER_OUT[1]
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.softmax_dlayer_in(LAYER_OUT[0], LAYER_OUT[1], OUT_BUFFER_TEMP[0], gpu_ind)
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
		
		OUT_BUFFER_TEMP = set_buffer(g, OUT_BUFFER_TEMP, gpu_ind)
	
	if len(args[0][1]) == 2:
		OUT_BUFFER_TEMP[1] = (dim1,dim2,dim1,dim2)
	else:
		OUT_BUFFER_TEMP[1] = (dim1,dim2,dim1,dim2,1)
	
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	t_main[1] += time.time() - t
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
		
		assert len(in_shape[0]) == 2 or (len(in_shape[0]) == 3 and in_shape[0][2] == 1)
		
		LAYERS[layer_ind]['forward_F'] = softmax
		LAYERS[layer_ind]['out_shape'] = in_shape[0][:2]
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = [in_source]
		LAYERS[layer_ind]['deriv_F'] = [softmax_dlayer_in]
		LAYERS[layer_ind]['in_prev'] = [False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None]]
		
		return layer_ind

