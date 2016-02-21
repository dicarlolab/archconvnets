import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0,0]

def random_function(size):
	return np.asarray(np.random.random(size) - .5, dtype='single')
	
def random_normal_function(size):
	return np.asarray(np.random.normal(loc=0, scale=.1, size=size), dtype='single')

def random_normal_bias_function(size):
	return np.asarray(np.random.normal(loc=0, scale=.01, size=size), dtype='single')

# additional_args[0]: Squeeze output or not, 
# additional_args[1] (sum_all): collapse x from [img,k,j] to [img,k*j,1] to give an output of [img,i,1] as opposed to [img,i,j]
def linear_F(args, OUT_BUFFER=None, additional_args=[True, False], gpu_ind=GPU_IND):
	t = time.time()
	
	squeeze, sum_all = additional_args
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	n_imgs = X[1][0]
	
	# if source is a conv layer (4D input), sum across everything
	if len(X[1]) != 3 or sum_all:
		X_reshaped = (n_imgs, np.prod(X[1][1:]), 1)
	else:
		X_reshaped = X[1]
	
	# reshape F into two dimensions:
	# (a,b,c,d,e) -> (a*b*c*d, e)
	F_reshaped = (np.prod(F[1][:-1]), F[1][-1])
	
	_ntm_module3.linear_F(F[0], F_reshaped, X[0], X_reshaped, OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = (n_imgs,) + F[1][:-1] + (X_reshaped[-1],)
	
	if squeeze and OUT_BUFFER[1][-1] == 1: # squeeze
		OUT_BUFFER[1] = OUT_BUFFER[1][:-1]
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

# additional_args = [True]: squeeze output last dimension
def linear_F_dx(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[True], gpu_ind=GPU_IND):
	t = time.time()
	
	squeeze, sum_all = additional_args
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	n_imgs = X[1][0]
	
	# if source is a conv layer (4D input), sum across everything
	if len(X[1]) != 3 or sum_all:
		X_reshaped = (n_imgs, np.prod(X[1][1:]), 1)
	else:
		X_reshaped = X[1]
	
	# reshape buffer1 into two dimensions:
	# (a,b,c,d,e) -> (a*b*c*d, e)
	F_reshaped = (np.prod(F[1][:-1]), F[1][-1])
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	dim_above = np.prod(DERIV_ABOVE[1][1:1+n_dim_not_summed])
	DERIV_ABOVE_reshaped = (n_imgs, dim_above) + DERIV_ABOVE[1][n_dim_not_summed+1:]
	
	# so we have deriv_above (3d: i,j,k) and x (2d: k,l), compute batched dot product (i,j,l)
	_ntm_module3.linear_F_dx(F[0], F_reshaped, X_reshaped, DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed+1] + X[1][1:]
	
	check_buffer(OUT_BUFFER)
	
	t_main[1] += time.time() - t
	return OUT_BUFFER

# additional_args = [True]: squeeze output last dimension
def linear_F_dF(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[True], gpu_ind=GPU_IND):
	t = time.time()
	
	squeeze, sum_all = additional_args
	F, X = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	n_imgs = X[1][0]
	
	# if source is a conv layer (4D input), sum across everything
	if len(X[1]) != 3 or sum_all:
		X_reshaped = (n_imgs, np.prod(X[1][1:]), 1)
	else:
		X_reshaped = X[1]
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	dim_above = np.prod(DERIV_ABOVE[1][1:1+n_dim_not_summed])
	
	if X[1][-1] != 1:
		DERIV_ABOVE_reshaped = (n_imgs, dim_above) + DERIV_ABOVE[1][n_dim_not_summed+1:]
	else:
		DERIV_ABOVE_reshaped = (n_imgs, dim_above, np.prod(F[1][:-1]), 1)
	
	# now: dot(deriv_above, x.T)
	_ntm_module3.linear_F_dF(X[0], X_reshaped, DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], gpu_ind)
	
	# did we sum all the images together or not?
	OUT_BUFFER[1] = F[1]
	if dim_above != 1:
		OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed+1] + F[1]
	
	check_buffer(OUT_BUFFER)
	
	t_main[2] += time.time() - t
	return OUT_BUFFER

def add_linear_F_layer(LAYERS, name, n_filters, source=None, sum_all=False, squeeze=True, random_function=random_normal_function, init=0):
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
		
		#############
		# source for X ( in_shape[1] ):
		
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
		
		
		#################
		# reshape X and use the new shape to determine the shape of F and the output shape
		assert len(in_shape[1]) > 2, "X should be >= 3 dim"
		
		n_imgs = in_shape[1][0]
		# if source is a conv layer (4D input), sum across everything
		if len(in_shape[1]) != 3 or sum_all:
			X_reshaped = (n_imgs, np.prod(in_shape[1][1:]), 1)
		else:
			X_reshaped = in_shape[1]
		
		##############
		# determine F and output shapes
		
		# if n_filters is an int or a tuple
		if isinstance(n_filters,int):
			in_shape[0] = (n_filters, X_reshaped[1])
			out_shape = (n_imgs, n_filters, X_reshaped[-1])
		else:
			in_shape[0] = n_filters + (X_reshaped[1],)
			out_shape = (n_imgs,) + in_shape[0][:-1] + (X_reshaped[-1],)
			
		######
		# squeeze
		if squeeze and out_shape[-1] == 1:
			out_shape = out_shape[:-1]
		
		LAYERS[layer_ind]['forward_F'] = linear_F
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = [random_function, in_source]
		LAYERS[layer_ind]['deriv_F'] = [linear_F_dF, linear_F_dx]
		LAYERS[layer_ind]['in_prev'] = [False, in_prev1]
		LAYERS[layer_ind]['additional_forward_args'] = [squeeze, sum_all]
		LAYERS[layer_ind]['additional_deriv_args'] = [[squeeze, sum_all], [squeeze, sum_all]]
		
		return layer_ind
		
