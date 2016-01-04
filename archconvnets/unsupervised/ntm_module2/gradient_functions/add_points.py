import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

# unlike point_wise_add, this defaults to storing the output in a new buffer instead of overwriting the first argument
def add_points(args, OUT_BUFFER=None, scalar=1, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	A, B = args
	check_buffer(A)
	check_buffer(B)
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = copy.deepcopy(A[1])
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		_ntm_module2.point_wise_add(A[0], B[0], np.single(scalar), OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		A_local = return_buffer(A,gpu_ind)
		B_local = return_buffer(B,gpu_ind)
		OUT_BUFFER = set_buffer(A_local + B_local*scalar, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = copy.deepcopy(B[1])
	return OUT_BUFFER

def add_points_dinput(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	A, B = args
	check_buffer(A)
	check_buffer(B)
	check_buffer(LAYER_OUT)
	assert A[1] == B[1]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module2.add_points_dinput(A[1], OUT_BUFFER[0], gpu_ind)
	else:
		######### CPU
		out = np.zeros(np.concatenate((args[0][1], args[0][1])), dtype='single')
		for i in range(out.shape[0]):
			out[i,range(out.shape[1]),i,range(out.shape[1])] = 1
		OUT_BUFFER = set_buffer(out, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((A[1], A[1])))
	return OUT_BUFFER
	
def add_add_layer(LAYERS, name, source):
	assert isinstance(name, str)
	assert isinstance(source, list)
	assert len(source) == 2
	assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
	
	source_A = find_layer(LAYERS, source[0])
	out_shape = LAYERS[source_A]['out_shape']
	if source[1] == name: # input is itself
		source_B = len(LAYERS)
		assert source_A != source_B
	else:
		source_B = find_layer(LAYERS, source[1])
		assert out_shape == LAYERS[source_B]['out_shape']
	
	LAYERS.append({ 'name': name, 'forward_F': add_points, \
				'out_shape': out_shape, \
				'in_shape': [out_shape, out_shape], \
				'in_source': [source_A, source_B], \
				'deriv_F': [add_points_dinput, add_points_dinput] })
	
	check_network(LAYERS)
	return LAYERS