import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

# c = a*b     a and b must be of the same dimensionality
def mult_points(args, OUT_BUFFER=None, additional_args=[], gpu_ind=0):
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
		_ntm_module2.mult_points(A[0], B[0], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		A_local = return_buffer(A,gpu_ind)
		B_local = return_buffer(B,gpu_ind)
		OUT_BUFFER = set_buffer(A_local*B_local, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = copy.deepcopy(B[1])
	return OUT_BUFFER

# additional_args == 0: deriv. wrt A
# additional_args == 1: deriv. wrt B
def mult_points_dinput(args, LAYER_OUT, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(additional_args) == 1
	assert additional_args[0] == 0 or additional_args[0] == 1
	A, B = args
	check_buffer(A)
	check_buffer(B)
	check_buffer(LAYER_OUT)
	assert A[1] == B[1]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		if additional_args[0] == 0: # deriv. wrt A
			_ntm_module2.mult_points_dinput(B[0], A[1], OUT_BUFFER[0], gpu_ind)
		else: # deriv. wrt B
			_ntm_module2.mult_points_dinput(A[0], A[1], OUT_BUFFER[0], gpu_ind)
	else:
		######### CPU
		if additional_args[0] == 0: # deriv. wrt A
			a = return_buffer(B,gpu_ind)
		else: # deriv. wrt B
			a = return_buffer(A,gpu_ind)
		out = np.zeros(np.concatenate((args[0][1], args[0][1])), dtype='single')
		for i in range(out.shape[0]):
			out[i,range(out.shape[1]),i,range(out.shape[1])] = a[i]
		OUT_BUFFER = set_buffer(out, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((A[1], A[1])))
	return OUT_BUFFER
	
# c = a*b
def add_mult_layer(LAYERS, name, source):
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
	
	LAYERS.append({ 'name': name, 'forward_F': mult_points, \
				'out_shape': out_shape, \
				'in_shape': [out_shape, out_shape], \
				'in_source': [source_A, source_B], \
				'deriv_F': [mult_points_dinput, mult_points_dinput], \
				'additional_forward_args': [],\
				'additional_deriv_args': [[0], [1]]})
	
	check_network(LAYERS)
	return len(LAYERS)-1