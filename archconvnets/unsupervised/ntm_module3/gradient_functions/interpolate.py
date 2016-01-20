import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0,0,0]

# print interp_gate_out.shape, o_content.shape, o_prev.shape
# (16, 1) (16, 6) (16, 6)
def interpolate(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	INTERP_GATE_OUT, O_CONTENT, O_PREV = args
	
	check_buffer(INTERP_GATE_OUT)
	check_buffer(O_CONTENT)
	check_buffer(O_PREV)
	assert len(INTERP_GATE_OUT[1]) == len(O_CONTENT[1]) == len(O_PREV[1]) == 2
	assert INTERP_GATE_OUT[1][0] == O_CONTENT[1][0] == O_PREV[1][0]
	assert O_CONTENT[1][1] == O_PREV[1][1]
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		_ntm_module3.interpolate(INTERP_GATE_OUT[0], O_CONTENT[0], O_PREV[0], O_CONTENT[1], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		interp_gate_out = return_buffer(INTERP_GATE_OUT,gpu_ind)
		o_content = return_buffer(O_CONTENT,gpu_ind)
		o_prev = return_buffer(O_PREV,gpu_ind)
		
		OUT_BUFFER = set_buffer(interp_gate_out * o_content + (1 - interp_gate_out) * o_prev, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = copy.deepcopy(O_CONTENT[1])
	check_buffer(OUT_BUFFER)
	t_main[0] += time.time() - t
	return OUT_BUFFER

def interpolate_dinterp_gate_out(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None],gpu_ind=GPU_IND):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	INTERP_GATE_OUT, O_CONTENT, O_PREV = args
	
	check_buffer(INTERP_GATE_OUT)
	check_buffer(O_CONTENT)
	check_buffer(O_PREV)
	check_buffer(DERIV_ABOVE)
	assert len(INTERP_GATE_OUT[1]) == len(O_CONTENT[1]) == len(O_PREV[1]) == 2
	assert INTERP_GATE_OUT[1][0] == O_CONTENT[1][0] == O_PREV[1][0]
	assert O_CONTENT[1][1] == O_PREV[1][1]
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.interpolate_dinterp_gate_out(O_CONTENT[0], O_CONTENT[1], O_PREV[0], OUT_BUFFER_TEMP[0], gpu_ind)
	else: 
		############ CPU
		interp_gate_out = return_buffer(INTERP_GATE_OUT,gpu_ind)
		o_content = return_buffer(O_CONTENT,gpu_ind)
		o_prev = return_buffer(O_PREV,gpu_ind)
		
		temp = o_content - o_prev
		temp2 = np.zeros((temp.shape[0], temp.shape[1], interp_gate_out.shape[0], 1),dtype='single')
		
		temp2[range(temp2.shape[0]), :, range(temp2.shape[0])] = temp[:,:,np.newaxis]
		
		OUT_BUFFER_TEMP = set_buffer(temp2, OUT_BUFFER_TEMP, gpu_ind)
	
	OUT_BUFFER_TEMP[1] = tuple(np.concatenate((O_CONTENT[1], INTERP_GATE_OUT[1])))
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	t_main[1] += time.time() - t
	return OUT_BUFFER
	
def interpolate_do_content(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	INTERP_GATE_OUT, O_CONTENT, O_PREV = args
	
	check_buffer(INTERP_GATE_OUT)
	check_buffer(O_CONTENT)
	check_buffer(O_PREV)
	check_buffer(DERIV_ABOVE)
	assert len(INTERP_GATE_OUT[1]) == len(O_CONTENT[1]) == len(O_PREV[1]) == 2
	assert INTERP_GATE_OUT[1][0] == O_CONTENT[1][0] == O_PREV[1][0]
	assert O_CONTENT[1][1] == O_PREV[1][1]
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.interpolate_do_content(INTERP_GATE_OUT[0], O_PREV[1], OUT_BUFFER_TEMP[0], gpu_ind)
	else: 
		############ CPU
		interp_gate_out = return_buffer(INTERP_GATE_OUT,gpu_ind)
		o_content = return_buffer(O_CONTENT,gpu_ind)
		o_prev = return_buffer(O_PREV,gpu_ind)
		
		temp = interp_gate_out
		n = o_content.shape[1]
		temp2 = np.zeros((o_content.shape[0], n, o_content.shape[0], n),dtype='single')
		
		for i in range(temp2.shape[0]):
			temp2[i,range(n),i,range(n)] = temp[i]
				
		OUT_BUFFER_TEMP = set_buffer(temp2, OUT_BUFFER_TEMP, gpu_ind)
	
	OUT_BUFFER_TEMP[1] = tuple(np.concatenate((O_CONTENT[1], O_CONTENT[1])))
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	t_main[2] += time.time() - t
	return OUT_BUFFER
	
def interpolate_do_prev(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	INTERP_GATE_OUT, O_CONTENT, O_PREV = args
	
	check_buffer(INTERP_GATE_OUT)
	check_buffer(O_CONTENT)
	check_buffer(O_PREV)
	check_buffer(DERIV_ABOVE)
	assert len(INTERP_GATE_OUT[1]) == len(O_CONTENT[1]) == len(O_PREV[1]) == 2
	assert INTERP_GATE_OUT[1][0] == O_CONTENT[1][0] == O_PREV[1][0]
	assert O_CONTENT[1][1] == O_PREV[1][1]
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.interpolate_do_prev(INTERP_GATE_OUT[0], O_PREV[1], OUT_BUFFER_TEMP[0], gpu_ind)
	else: 
		############ CPU
		interp_gate_out = return_buffer(INTERP_GATE_OUT,gpu_ind)
		o_content = return_buffer(O_CONTENT,gpu_ind)
		o_prev = return_buffer(O_PREV,gpu_ind)
		
		temp = 1 - interp_gate_out
		n = o_prev.shape[1]
		temp2 = np.zeros((o_prev.shape[0], n, o_prev.shape[0], n),dtype='single')
		
		for i in range(temp2.shape[0]):
			temp2[i,range(n),i,range(n)] = temp[i]
				
		OUT_BUFFER_TEMP = set_buffer(temp2, OUT_BUFFER_TEMP, gpu_ind)
	
	OUT_BUFFER_TEMP[1] = tuple(np.concatenate((O_CONTENT[1], O_PREV[1])))
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	t_main[3] += time.time() - t
	return OUT_BUFFER

def add_interpolate_layer(LAYERS, name, source, init=0):
	assert isinstance(name, str)
	assert isinstance(source, list)
	assert len(source) == 3
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		arg2_prev = source[2][-1] == '-'
		
		in_shape = [None]*3
		
		for arg in range(3):
			source[arg] = find_layer(LAYERS, source[arg])
			assert source[arg] is not None, 'could not find source layer %i' % arg
			
		in_shape[0] = LAYERS[source[0]]['out_shape']
		in_shape[1] = LAYERS[source[1]]['out_shape']
		in_shape[2] = LAYERS[source[1]]['out_shape']
		
		out_shape = LAYERS[source[1]]['out_shape']
		
		assert len(out_shape) == len(LAYERS[source[1]]['out_shape']) == 2
		assert out_shape == LAYERS[source[1]]['out_shape']
		assert out_shape[0] == LAYERS[source[0]]['out_shape'][0]
		assert LAYERS[source[0]]['out_shape'][1] == 1
		
		LAYERS[layer_ind]['forward_F']= interpolate
		LAYERS[layer_ind]['out_shape'] = LAYERS[source[1]]['out_shape']
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = source
		LAYERS[layer_ind]['deriv_F'] = [interpolate_dinterp_gate_out, interpolate_do_content, interpolate_do_prev]
		LAYERS[layer_ind]['in_prev'] = [False, False, arg2_prev]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None], [None], [None]]
		
		return layer_ind