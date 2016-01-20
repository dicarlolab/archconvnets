import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0]

# c = a*b     a and b must be of the same dimensionality
def mult_points(args, OUT_BUFFER=None, additional_args=[], gpu_ind=0):
	t = time.time()
	assert len(additional_args) == 0
	assert isinstance(gpu_ind,int)
	A, B = args
	check_buffer(A)
	check_buffer(B)
	assert A[1] == B[1]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module3.mult_points(A[0], B[0], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		A_local = return_buffer(A,gpu_ind)
		B_local = return_buffer(B,gpu_ind)
		OUT_BUFFER = set_buffer(A_local*B_local, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = copy.deepcopy(B[1])
	t_main[0] += time.time() - t
	return OUT_BUFFER

# additional_args == 0: deriv. wrt A
# additional_args == 1: deriv. wrt B
def mult_points_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert len(additional_args) == 1
	assert additional_args[0] == 0 or additional_args[0] == 1
	A, B = args
	check_buffer(A)
	check_buffer(B)
	check_buffer(LAYER_OUT)
	check_buffer(DERIV_ABOVE)
	assert A[1] == B[1]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		if additional_args[0] == 0: # deriv. wrt A
			_ntm_module3.mult_points_dinput(B[0], A[1], OUT_BUFFER_TEMP[0], gpu_ind)
		else: # deriv. wrt B
			_ntm_module3.mult_points_dinput(A[0], A[1], OUT_BUFFER_TEMP[0], gpu_ind)
	else:
		######### CPU
		if additional_args[0] == 0: # deriv. wrt A
			a = return_buffer(B,gpu_ind)
		else: # deriv. wrt B
			a = return_buffer(A,gpu_ind)
		out = np.zeros(np.concatenate((args[0][1], args[0][1])), dtype='single')
		for i in range(out.shape[0]):
			out[i,range(out.shape[1]),i,range(out.shape[1])] = a[i]
		OUT_BUFFER_TEMP = set_buffer(out, OUT_BUFFER_TEMP, gpu_ind)
	
	OUT_BUFFER_TEMP[1] = tuple(np.concatenate((A[1], A[1])))
	
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	t_main[1] += time.time() - t
	return OUT_BUFFER
	
# c = a*b
def add_mult_layer(LAYERS, name, source, init=0):
	assert isinstance(name, str)
	assert len(source) == 2
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		in_prev = [None]*2
		in_prev[0] = False
		in_prev[1] = source[1][-1] == '-'
		
		source_A = find_layer(LAYERS, source[0])
		out_shape = LAYERS[source_A]['out_shape']
		
		source_B = find_layer(LAYERS, source[1])
		if in_prev[1] == False:
			assert out_shape == LAYERS[source_B]['out_shape']
		
		LAYERS[layer_ind]['forward_F'] = mult_points
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = [out_shape, out_shape]
		LAYERS[layer_ind]['in_source'] = [source_A, source_B]
		LAYERS[layer_ind]['deriv_F'] = [mult_points_dinput, mult_points_dinput]
		LAYERS[layer_ind]['additional_forward_args'] = []
		LAYERS[layer_ind]['additional_deriv_args'] = [[0], [1]]
		LAYERS[layer_ind]['in_prev'] = in_prev
		
		return layer_ind