import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *

# c = a + scalar*b
# unlike point_wise_add, this defaults to storing the output in a new buffer instead of overwriting the first argument
def add_points(args, OUT_BUFFER=None, additional_args=[1], gpu_ind=0):
	assert len(additional_args) == 1
	assert isinstance(gpu_ind,int)
	scalar = additional_args[0]
	A, B = args
	check_buffer(A)
	check_buffer(B)
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = copy.deepcopy(A[1])
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		_ntm_module3.point_wise_add(A[0], B[0], np.single(scalar), np.single(1), OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		A_local = return_buffer(A,gpu_ind)
		B_local = return_buffer(B,gpu_ind)
		OUT_BUFFER = set_buffer(A_local + B_local*scalar, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = copy.deepcopy(B[1])
	return OUT_BUFFER

def add_points_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[1], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(additional_args) == 1
	scalar = additional_args[0]
	A, B = args
	check_buffer(A)
	check_buffer(B)
	check_buffer(LAYER_OUT)
	check_buffer(DERIV_ABOVE)
	assert A[1] == B[1]
	#assert len(A[1]) == 2 or len(A[1]) == 4
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.add_points_dinput((np.prod(A[1]), 1), OUT_BUFFER_TEMP[0], np.single(scalar), gpu_ind)
	else:
		######### CPU
		out = np.zeros(np.concatenate((args[0][1], args[0][1])), dtype='single')
		for i in range(out.shape[0]):
			out[i,range(out.shape[1]),i,range(out.shape[1])] = scalar
		OUT_BUFFER_TEMP = set_buffer(out, OUT_BUFFER_TEMP, gpu_ind)
	
	OUT_BUFFER_TEMP[1] = tuple(np.concatenate((A[1], A[1])))
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	
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

	