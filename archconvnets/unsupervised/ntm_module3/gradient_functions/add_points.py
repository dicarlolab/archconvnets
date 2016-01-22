import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0]

# c = a + scalar*b
# unlike point_wise_add, this defaults to storing the output in a new buffer instead of overwriting the first argument
def add_points(args, OUT_BUFFER=None, additional_args=[1], gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = copy.deepcopy(A[1])
	else:
		OUT_BUFFER = init_buffer()
	
	_ntm_module3.point_wise_add(A[0], B[0], additional_args[0], np.single(1), OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = copy.deepcopy(B[1])
	
	if DEBUG:
		check_buffer(A)
		check_buffer(B)
		assert len(additional_args) == 1
		assert isinstance(gpu_ind,int)
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

# c = a + scalar*b
# additional_args[0] denotes the scalar (when equal to 1, the deriv wrt to a is taken [and b if scalar=1])
def add_points_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[1], gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	_ntm_module3.add_points_dinput((np.prod(A[1]), 1), OUT_BUFFER[0], DERIV_ABOVE[0], additional_args[0], gpu_ind)
	
	# reshape back to original dimensions
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dim_not_summed], A[1])))
	
	if DEBUG:
		assert isinstance(gpu_ind,int)
		assert len(additional_args) == 1
		check_buffer(A)
		check_buffer(B)
		check_buffer(LAYER_OUT)
		check_buffer(DERIV_ABOVE)
		assert A[1] == B[1]
		#assert len(A[1]) == 2 or len(A[1]) == 4
		check_buffer(OUT_BUFFER)
		
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

	