import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0]

# c = a + scalar*b
# unlike point_wise_add, this defaults to storing the output in a new buffer instead of overwriting the first argument
def add_points(args, OUT_BUFFER=None, additional_args=[1], gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	_ntm_module3.add_points(A[0], B[0], additional_args[0], 1., OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = B[1]
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

# out_buffer[img] = a[img] + b
def add_points_batch(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	_ntm_module3.add_points_batch(A[0], B[0], OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = A[1]
	
	t_main[0] += time.time() - t
	return OUT_BUFFER	


# out_buffer[img] = a[img] + b

# deriv_above: (2,3, 10, 4,5),   a: (10, 4,5)
# b: (4,5) ===> return (2,3, 4,5) (sum deriv_above across images):
#	deriv_above(2*3, 10, 4*5) batch first dimension -> transpose -> deriv_above*(1,1)

# NOTE: add_points_batch_dinputA = add_points_dinput [with additional_args=[1]]
def add_points_batch_dinputB(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	n_imgs = LAYER_OUT[1][0]
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	dim_above = np.prod(DERIV_ABOVE[1][1:1+n_dim_not_summed])
	DERIV_ABOVE_reshaped = (n_imgs, dim_above) + DERIV_ABOVE[1][n_dim_not_summed+1:]
	
	_ntm_module3.add_points_batch_dinputB(A[0], B[0], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], gpu_ind)
	# computes:
	# set_buffer(return_buffer(DERIV_ABOVE).sum(0).sum(0), OUT_BUFFER)
	
	OUT_BUFFER[1] = B[1]
	if dim_above != 1:
		OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed+1] + B[1]
	
	t_main[1] += time.time() - t
	return OUT_BUFFER	

# c = a + scalar*b
# additional_args[0] denotes the scalar (when equal to 1, the deriv wrt to a is taken [and b if scalar=1])
def add_points_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[1], gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	_ntm_module3.add_points_dinput(OUT_BUFFER[0], DERIV_ABOVE[0], additional_args[0], gpu_ind)
	
	OUT_BUFFER[1] = DERIV_ABOVE[1]
	
	# deriv_above (1, 1) ERR
	# deriv_above (1, 4, 16, 8) F3s (where 4 is the image dim)
	# then move the image dim to the front.
	n_imgs = LAYER_OUT[1][0]
	if n_imgs != DERIV_ABOVE[1][0] and DERIV_ABOVE[1][0] == 1:
		OUT_BUFFER[1] = (n_imgs, 1) + LAYER_OUT[1][1:]
	
	t_main[1] += time.time() - t
	return OUT_BUFFER
	
	
# c = a + scalar*b
def add_add_layer(LAYERS, name, source, scalar=1, init=0):
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
		
		in_prev = [None]*2
		
		source_A = find_layer(LAYERS, source[0])
		in_prev[0] = source[0][-1] == '-'
		
		if isinstance(source[1], str):
			in_prev[1] = source[1][-1] == '-'
			source_B = find_layer(LAYERS, source[1])
		else:
			in_prev[1] = False
			source_B = source[1]
			assert isinstance(source_B,int)
		
		assert source_A is not None, 'could not find layer %s' % source[0]
		assert source_B is not None, 'could not find layer %s' % source[1]
		
		assert source_A != source_B
		
		if in_prev[0]:
			out_shape = LAYERS[source_B]['out_shape']
		else:
			out_shape = LAYERS[source_A]['out_shape']
		
		if np.sum(in_prev) == 0 and source_B != -1:
			assert LAYERS[source_B]['out_shape'] == LAYERS[source_A]['out_shape']
		
		LAYERS[layer_ind]['forward_F'] = add_points
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = [out_shape, out_shape]
		LAYERS[layer_ind]['in_source'] = [source_A, source_B]
		LAYERS[layer_ind]['deriv_F'] = [add_points_dinput, add_points_dinput]
		LAYERS[layer_ind]['additional_forward_args'] = [scalar]
		LAYERS[layer_ind]['additional_deriv_args'] = [[1], [scalar]]
		LAYERS[layer_ind]['in_prev'] = in_prev
		
		return layer_ind

	
