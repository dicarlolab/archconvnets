import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

# additional_args= [PAD]
def conv(args, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, IMGS = args
	check_buffer(F)
	check_buffer(IMGS)
	n_filters, n_channels, filter_sz, filter_sz2  = F[1]
	n_imgs, n_channels2, img_sz, img_sz2 = IMGS[1]
	PAD = additional_args[0]
	assert isinstance(PAD,int)
	assert PAD >= 0
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert IMGS[1][0] == 1
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		OUT_BUFFER[1] = _ntm_module2.conv(F[0], F[1], IMGS[0], IMGS[1], PAD, OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		assert False, 'cpu conv not supported'
		
	return OUT_BUFFER

def conv_ddata(args, LAYER_OUT, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, IMGS = args
	check_buffer(F)
	check_buffer(IMGS)
	n_filters, n_channels, filter_sz, filter_sz2  = F[1]
	n_imgs, n_channels2, img_sz, img_sz2 = IMGS[1]
	PAD = additional_args[0]
	assert isinstance(PAD,int)
	assert PAD >= 0
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert IMGS[1][0] == LAYER_OUT[1][0]
	assert F[1][0] == LAYER_OUT[1][1]
	assert IMGS[1][0] == 1
	
	######## use identity matrix so we get derivs wrt to each output location
	layer_out_shape = LAYER_OUT[1]
	
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
		OUT_BUFFER[1] = _ntm_module2.conv_ddata(F[0], F[1], IMGS[0], IMGS[1], DERIV_ABOVE[0], DERIV_ABOVE[1], PAD, OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		assert False, 'cpu conv not supported'
	
	free_buffer(DERIV_ABOVE)
	
	OUT_BUFFER[1] = tuple(np.concatenate((LAYER_OUT[1], IMGS[1])))
	
	return OUT_BUFFER
	
def conv_dfilter(args, LAYER_OUT, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, IMGS = args
	check_buffer(F)
	check_buffer(IMGS)
	n_filters, n_channels, filter_sz, filter_sz2  = F[1]
	n_imgs, n_channels2, img_sz, img_sz2 = IMGS[1]
	PAD = additional_args[0]
	assert isinstance(PAD,int)
	assert PAD >= 0
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert IMGS[1][0] == LAYER_OUT[1][0]
	assert F[1][0] == LAYER_OUT[1][1]
	assert IMGS[1][0] == 1
	
	######## use identity matrix so we get derivs wrt to each output location
	layer_out_shape = LAYER_OUT[1]
	
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
		OUT_BUFFER[1] = _ntm_module2.conv_dfilter(F[0], F[1], IMGS[0], IMGS[1], DERIV_ABOVE[0], DERIV_ABOVE[1], PAD, OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		assert False, 'cpu conv not supported'
	
	free_buffer(DERIV_ABOVE)
	
	OUT_BUFFER[1] = tuple(np.concatenate((LAYER_OUT[1], F[1])))
	
	return OUT_BUFFER
	
'''
def add_sharpen_layer(LAYERS, name, source, init=0):
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
		
		in_shape = [None]*2
		
		source[0] = find_layer(LAYERS, source[0])
		assert source[0] is not None, 'could not find source layer 0'
		
		if isinstance(source[1],int) != True and source[1] != -1:
			source[1] = find_layer(LAYERS, source[1])
		
		in_shape[0] = LAYERS[source[0]]['out_shape']
		in_shape[1] = (in_shape[0][0], 1)
		
		LAYERS[layer_ind]['forward_F'] = sharpen
		LAYERS[layer_ind]['out_shape'] = LAYERS[source[0]]['out_shape']
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = source
		LAYERS[layer_ind]['deriv_F'] = [sharpen_dw, sharpen_dgamma]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		
		return layer_ind
'''