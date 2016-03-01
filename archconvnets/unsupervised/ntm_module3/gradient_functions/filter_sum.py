import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0,0]

def filter_sum(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	f = return_buffer(F)
	x = return_buffer(X)
	
	# X: [n_imgs, n_channels, img_sz, img_sz]
	# F: [n_channels_out, n_channels, img_sz, img_sz]
	
	out = np.einsum(f, range(4), x, [4,1,2,3], [4,0,2,3])
	OUT_BUFFER = set_buffer(out, OUT_BUFFER)
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

def filter_sum_dX(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	deriv_above = return_buffer(DERIV_ABOVE)
	
	# collapse above dims (n_imgs, dim_above, n_channels_out, sz, sz) -> (n_imgs, dims, n_channels_out, sz, sz)
	deriv_above = deriv_above.reshape((deriv_above.shape[0], np.prod(deriv_above.shape[1:-3])) + deriv_above.shape[-3:])
	
	# F: [n_channels_out, n_channels, img_sz, img_sz]
	f = return_buffer(F)
	
	out = np.einsum(deriv_above, range(5), f, [2, 5, 3, 4], [0, 1, 5, 3, 4])
	
	OUT_BUFFER = set_buffer(out, OUT_BUFFER)
	
	OUT_BUFFER[1] = deriv_above.shape[:-3] + X[1][1:]
	check_buffer(OUT_BUFFER)
	
	t_main[1] += time.time() - t
	return OUT_BUFFER

def filter_sum_dF(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	deriv_above = return_buffer(DERIV_ABOVE)
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	dim_above = np.prod(DERIV_ABOVE[1][1:1+n_dim_not_summed])
	n_imgs = deriv_above.shape[0]
	
	# collapse above dims (n_imgs, dim_above, n_channels_out, sz, sz) -> (n_imgs, dims, n_channels_out, sz, sz)
	deriv_above = deriv_above.reshape((n_imgs, dim_above) + deriv_above.shape[-3:])
	
	# X: [n_imgs, n_channels, img_sz, img_sz]
	x = return_buffer(X)
	
	if dim_above != 1: # don't sum imgs/dim_above
		out = np.einsum(deriv_above, range(5), x, [0, 5,3,4], [0, 1, 2, 5, 3, 4])
		OUT_BUFFER = set_buffer(out, OUT_BUFFER)
		OUT_BUFFER[1] = deriv_above.shape[:-3] + F[1]
	else:
		out = np.einsum(deriv_above, range(5), x, [0, 5,3,4], [2,5,3,4])
		OUT_BUFFER = set_buffer(out, OUT_BUFFER)
		OUT_BUFFER[1] = F[1]
	
	check_buffer(OUT_BUFFER)
	
	t_main[2] += time.time() - t
	return OUT_BUFFER

def add_filter_sum_layer(LAYERS, name, n_channels_out, source=None, random_function=random_normal_function, init=0):
	assert isinstance(name, str)
	assert isinstance(n_channels_out, int)
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		in_shape = [None]*2
		
		#############
		# source for X ( in_shape[1] ):
		
		# default to previous layer as input
		if source is None:
			in_source = layer_ind-1
		# find layer specified
		elif isinstance(source,str):
			in_source = find_layer(LAYERS, source)
			assert in_source is not None, 'could not find source layer %i' % source
			assert source[-1] != '-'
		else:
			assert False, 'unknown source input'
		
		in_shape[1] = LAYERS[in_source]['out_shape']
		assert len(in_shape[1]) == 4, "X should be 4 dim"
		
		##############
		# determine F and output shapes
		n_imgs, n_channels, img_sz, img_sz2 = in_shape[1]
		assert img_sz == img_sz2
		
		in_shape[0] = (n_channels_out, n_channels, img_sz, img_sz)
		out_shape = (n_imgs, n_channels_out, img_sz, img_sz)
		
		LAYERS[layer_ind]['forward_F'] = filter_sum
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = [random_function, in_source]
		LAYERS[layer_ind]['deriv_F'] = [filter_sum_dF, filter_sum_dX]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None], [None]]
		
		return layer_ind
		
