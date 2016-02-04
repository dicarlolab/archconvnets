import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0]

# additional_args[0]: batch_imgs (separate correlations batched on first dim)
def pearson(args, OUT_BUFFER=None, additional_args=[False], gpu_ind=GPU_IND):
	t = time.time()
	
	batch_imgs = additional_args[0]

	W1, W2 = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	if batch_imgs:
		n_imgs = W1[1][0]
	else:
		n_imgs = 1
	
	_ntm_module3.pearson(W1[0], W2[0], OUT_BUFFER[0], n_imgs, gpu_ind)
	
	OUT_BUFFER[1] = (n_imgs,)
	
	if DEBUG:
		assert GPU
		assert isinstance(gpu_ind,int)
		assert additional_args == [None]
		check_buffer(W1)
		check_buffer(W2)
		assert np.prod(W1[1]) == np.prod(W2[1])
		check_buffer(OUT_BUFFER)
		check_buffer(OUT_BUFFER)
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

# wrt additional_args[0] (either 0 for w1 or 1 for w2)
# additional_args[1]: batch_imgs (separate correlations batched on first dim)
def pearson_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[0,False], gpu_ind=GPU_IND):
	t = time.time()
	
	deriv_wrt, batch_imgs = additional_args
	
	W1, W2 = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	if batch_imgs:
		n_imgs = W1[1][0]
	else:
		n_imgs = 1
	
	n_batches = np.prod(DERIV_ABOVE[1])/n_imgs
	
	if deriv_wrt == 0:
		_ntm_module3.pearson_dinput(W2[0], W1[0], OUT_BUFFER[0], DERIV_ABOVE[0], n_imgs, n_batches, gpu_ind)
	elif deriv_wrt == 1:
		_ntm_module3.pearson_dinput(W1[0], W2[0], OUT_BUFFER[0], DERIV_ABOVE[0], n_imgs, n_batches, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:len(DERIV_ABOVE[1])-len(LAYER_OUT[1])], W2[1])))
	
	if DEBUG:
		assert GPU
		assert isinstance(gpu_ind,int)
		assert len(additional_args) == 1
		assert deriv_wrt == 0 or deriv_wrt == 1
		check_buffer(OUT_BUFFER)
		check_buffer(W1)
		check_buffer(W2)
		check_buffer(DERIV_ABOVE)
		check_buffer(LAYER_OUT)
		assert np.prod(W1[1]) == np.prod(W2[1])
		assert np.prod(LAYER_OUT[1]) == 1
		assert DERIV_ABOVE[1][-1] == 1
	
	t_main[1] += time.time() - t
	return OUT_BUFFER

def add_pearson_layer(LAYERS, name, source, batch_imgs=False, init=0):
	assert isinstance(name, str)
	assert len(source) == 2
	assert isinstance(source[0],str)
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		in_source = [None]*2
		in_shape = [None]*2
		
		in_source[0] = find_layer(LAYERS, source[0])
		assert in_source[0] is not None, 'could not find source layer %i' % source[0]
		in_shape[0] = LAYERS[in_source[0]]['out_shape']
		
		if isinstance(source[1],str):
			in_source[1] = find_layer(LAYERS, source[1])
			in_shape[1] = LAYERS[in_source[1]]['out_shape']
			assert in_source[1] is not None, 'could not find source layer %i' % source[1]
		else:
			in_source[1] = source[1]
			in_shape[1] = in_shape[0]
		
		assert in_shape[0] == in_shape[1]
		
		if batch_imgs:
			out_shape = (in_shape[0][0],)
		else:
			out_shape = (1,)
		
		LAYERS[layer_ind]['forward_F'] = pearson
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = in_source
		LAYERS[layer_ind]['deriv_F'] = [pearson_dinput, pearson_dinput]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		LAYERS[layer_ind]['additional_forward_args'] = [batch_imgs]
		LAYERS[layer_ind]['additional_deriv_args'] = [[0, batch_imgs], [1, batch_imgs]]
		
		return layer_ind

