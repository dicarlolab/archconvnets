import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0]

# concat img channels
def concat(args, OUT_BUFFER=None, additional_args=[1], gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	a = return_buffer(A)
	b = return_buffer(B)
	
	out = np.ascontiguousarray(np.concatenate((a, b), axis=1))
	OUT_BUFFER = set_buffer(out, OUT_BUFFER)
	
	OUT_BUFFER[1] = (A[1][0], A[1][1] + B[1][1]) + A[1][2:]
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

# additional_args[0] denotes if deriv is taken to zeroth or first argument
def concat_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[1], gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	deriv_above = return_buffer(DERIV_ABOVE)
	
	n_chan_A = A[1][1]
	
	# collapse above dims (n_imgs, dim_above, n_channels, sz, sz) -> (dims, n_channels, sz, sz)
	deriv_above = deriv_above.reshape((np.prod(deriv_above.shape[:-3]),) + deriv_above.shape[-3:])
	
	if additional_args[0] == 0:
		deriv_above = deriv_above[:,:n_chan_A]
	else:
		deriv_above = deriv_above[:,n_chan_A:]
	
	# uncollapse
	deriv_above = np.ascontiguousarray(deriv_above.reshape(DERIV_ABOVE[1][:-3] + deriv_above.shape[1:]))
	
	OUT_BUFFER = set_buffer(deriv_above, OUT_BUFFER)
	
	t_main[1] += time.time() - t
	return OUT_BUFFER
	
# concat img channels
def add_concat_layer(LAYERS, name, source, init=0):
	assert isinstance(name, str)
	assert isinstance(source, list)
	assert len(source) == 2
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		assert isinstance(source[0], str)
		assert isinstance(source[1], str)
		
		source_A = find_layer(LAYERS, source[0])
		source_B = find_layer(LAYERS, source[1])
		
		assert source[0][-1] != '-', 'memory layer not supported... need to add mechanism to get channel count to this script...'
		assert source[1][-1] != '-', 'memory layer not supported... need to add mechanism to get channel count to this script...'
		
		assert source_A is not None, 'could not find layer %s' % source[0]
		assert source_B is not None, 'could not find layer %s' % source[1]
		
		assert source_A != source_B
		
		assert len(LAYERS[source_A]['out_shape']) == len(LAYERS[source_B]['out_shape']) == 4
		
		# everything except the channels needs to match:
		assert LAYERS[source_B]['out_shape'][0] == LAYERS[source_A]['out_shape'][0]
		assert LAYERS[source_B]['out_shape'][2:] == LAYERS[source_A]['out_shape'][2:]
		
		# sum channels:
		out_shape = LAYERS[source_A]['out_shape']
		out_shape = (out_shape[0],  out_shape[1] + LAYERS[source_B]['out_shape'][1]) + out_shape[2:]
		
		LAYERS[layer_ind]['forward_F'] = concat
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = [LAYERS[source_A]['out_shape'], LAYERS[source_B]['out_shape']]
		LAYERS[layer_ind]['in_source'] = [source_A, source_B]
		LAYERS[layer_ind]['deriv_F'] = [concat_dinput, concat_dinput]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[0], [1]]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		
		return layer_ind

	
