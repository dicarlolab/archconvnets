import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0,0]

def random_function(size):
	return np.asarray(np.random.random(size) - .5, dtype='single')
	
def random_normal_function(size):
	return np.asarray(np.random.normal(loc=0, scale=.01, size=size), dtype='single')

# additional_args = [True]: squeeze output last dimension
def linear_F_dx(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[True], gpu_ind=GPU_IND):
	t = time.time()
	
	squeeze, sum_all, batch_imgs = additional_args
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	# if source is a conv layer (4D input), sum across everything
	if len(X[1]) == 4 or sum_all:
		if batch_imgs:
			X_reshaped = (X[1][0], np.prod(X[1][1:]), 1)
		else:
			X_reshaped = (np.prod(X[1]), 1)
	else:
		X_reshaped = X[1]
	
	# reshape buffer1 into two dimensions:
	# (a,b,c,d,e) -> (a*b*c*d, e)
	F_reshaped = (np.prod(F[1][:len(F[1])-1]), F[1][-1])
	
	# reshape deriv_above to 3 dims, first of which we batch over
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	if batch_imgs:
		n_imgs = X[1][0]
		DERIV_ABOVE_reshaped = (np.prod(DERIV_ABOVE[1][:n_dim_not_summed+1]), DERIV_ABOVE[1][-2], DERIV_ABOVE[1][-1])
	else:
		n_imgs = 1
		DERIV_ABOVE_reshaped = (np.prod(DERIV_ABOVE[1][:n_dim_not_summed]), DERIV_ABOVE[1][-2], DERIV_ABOVE[1][-1])
	
	# so we have deriv_above (3d: i,j,k) and x (2d: k,l), compute batched dot product (i,j,l)
	_ntm_module3.linear_F_dx(F[0], F_reshaped, X_reshaped, DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], n_imgs, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dim_not_summed], X[1])))
	
	#print 'F_reshaped', F_reshaped, 'X_reshaped', X_reshaped, 'deriv_Above_reshaped', DERIV_ABOVE_reshaped, 'out_buffer', OUT_BUFFER[1]
	
	if DEBUG:
		check_buffer(F)
		check_buffer(X)
		check_buffer(LAYER_OUT)
		check_buffer(OUT_BUFFER)
		check_buffer(DERIV_ABOVE)
		assert len(F[1]) >= 2
		assert len(X[1]) == 2 or len(X[1]) == 4
		assert F[1][-1] == X_reshaped
	
	t_main[1] += time.time() - t
	return OUT_BUFFER

# additional_args = [True]: squeeze output last dimension
def linear_F_dF(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[True], gpu_ind=GPU_IND):
	t = time.time()
	
	squeeze, sum_all, batch_imgs = additional_args
	
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	
	# if source is a conv layer (4D input), sum across everything
	if len(X[1]) == 4 or sum_all:
		if batch_imgs:
			X_reshaped = (X[1][0], np.prod(X[1][1:]), 1)
		else:
			X_reshaped = (np.prod(X[1]), 1)
	else:
		X_reshaped = X[1]
	
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	
	if batch_imgs:
		n_batches = np.prod(DERIV_ABOVE[1][:n_dim_not_summed]) * LAYER_OUT[1][0] # deriv above * n_imgs
		# reshape deriv_above to 2 dims
		DERIV_ABOVE_reshaped = tuple(np.concatenate((np.asarray(n_batches)[np.newaxis], DERIV_ABOVE[1][n_dim_not_summed+1:])))
	else:
		n_batches = 1
		# reshape deriv_above to 2 dims
		if squeeze:
			DERIV_ABOVE_reshaped = (np.prod(DERIV_ABOVE[1]), 1)
		else:
			DERIV_ABOVE_reshaped = (np.prod(DERIV_ABOVE[1][:len(DERIV_ABOVE[1])-1]), DERIV_ABOVE[1][-1])
	
	#print 'F', F[1], 'X', X[1], 'x_reshaped', X_reshaped
	#print 'DERIV_ABOVE', DERIV_ABOVE[1], 'DERIV_ABOVE_reshaped', DERIV_ABOVE_reshaped
	
	# now: dot(deriv_above, x.T)
	_ntm_module3.linear_F_dF(X[0], X_reshaped, DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], n_batches, gpu_ind)
	
	# reshape back to original dimensions
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dim_not_summed], F[1])))
	
	#print 'out_buffer', OUT_BUFFER[1], 'X_reshaped', X_reshaped, 'deriv_above_reshaped', DERIV_ABOVE_reshaped, 'n_batches', n_batches
	
	if DEBUG:
		check_buffer(OUT_BUFFER)
		check_buffer(F)
		check_buffer(X)
		check_buffer(LAYER_OUT)
		check_buffer(OUT_BUFFER)
		check_buffer(DERIV_ABOVE)
		assert len(F[1]) >= 2
		assert len(X[1]) == 2 or len(X[1]) == 4
		assert F[1][-1] == X_reshaped[1][0]
		
	
	t_main[2] += time.time() - t
	return OUT_BUFFER

linear_F = dot

def add_linear_F_layer(LAYERS, name, n_filters, source=None, sum_all=False, squeeze=True, batch_imgs=False, random_function=random_function, init=0):
	assert isinstance(name, str)
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		in_shape = [None]*2
		in_prev1 = False
		
		# default to previous layer as input
		if source is None:
			in_source = layer_ind-1
			in_shape[1] = LAYERS[in_source]['out_shape']
		# find layer specified
		elif isinstance(source,str):
			in_source = find_layer(LAYERS, source)
			assert in_source is not None, 'could not find source layer %i' % source
			in_shape[1] = LAYERS[in_source]['out_shape']
			in_prev1 = source[-1] == '-'
		
		# input is user supplied
		elif isinstance(source,tuple):
			in_shape[1] = source
			in_source = -1
		else:
			assert False, 'unknown source input'
		
		if batch_imgs:
			n_batches = in_shape[1][0]
		else:
			n_batches = 1
		
		# if source is a conv layer (4D input), sum across everything
		if len(in_shape[1]) == 4 or sum_all:
			if batch_imgs:
				in_shape_reshaped = (np.prod(in_shape[1][1:]), 1)
			else:
				in_shape_reshaped = (np.prod(in_shape[1]), 1)
		else:
			if batch_imgs:
				in_shape_reshaped = copy.deepcopy(in_shape[1][1:])
			else:
				in_shape_reshaped = copy.deepcopy(in_shape[1])
		
		# if n_filters is an int or a tuple
		if isinstance(n_filters,int):
			in_shape[0] = (n_filters, in_shape_reshaped[0])
			if batch_imgs:
				out_shape = (n_batches, in_shape[0][0], in_shape_reshaped[1])
			else:
				out_shape = (in_shape[0][0], in_shape_reshaped[1])
		else:
			in_shape[0] = tuple(np.concatenate((np.asarray(n_filters), np.asarray(in_shape_reshaped[0])[np.newaxis])))
			if batch_imgs:
				out_shape = tuple(np.concatenate((np.asarray(n_batches)[np.newaxis], in_shape[0][:len(in_shape[0])-1], np.asarray(in_shape_reshaped[1])[np.newaxis])))
			else:
				out_shape = tuple(np.concatenate((in_shape[0][:len(in_shape[0])-1], np.asarray(in_shape_reshaped[1])[np.newaxis])))
		
		if squeeze and out_shape[-1] == 1:
			out_shape = out_shape[:len(out_shape)-1]
		
		LAYERS[layer_ind]['forward_F'] = linear_F
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = [random_function, in_source]
		LAYERS[layer_ind]['deriv_F'] = [linear_F_dF, linear_F_dx]
		LAYERS[layer_ind]['in_prev'] = [False, in_prev1]
		LAYERS[layer_ind]['additional_forward_args'] = [squeeze, sum_all, batch_imgs]
		LAYERS[layer_ind]['additional_deriv_args'] = [[squeeze, sum_all, batch_imgs], [squeeze, sum_all, batch_imgs]]
		
		return layer_ind
		
