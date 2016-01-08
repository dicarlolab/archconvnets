import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

# print interp_gate_out.shape, o_content.shape, o_prev.shape
# (16, 1) (16, 6) (16, 6)
def interpolate(args, OUT_BUFFER=None, scalar=1, gpu_ind=0):
	assert isinstance(gpu_ind,int)
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
		_ntm_module2.interpolate(INTERP_GATE_OUT[0], O_CONTENT[0], O_PREV[0], O_CONTENT[1], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		interp_gate_out = return_buffer(INTERP_GATE_OUT,gpu_ind)
		o_content = return_buffer(O_CONTENT,gpu_ind)
		o_prev = return_buffer(O_PREV,gpu_ind)
		
		OUT_BUFFER = set_buffer(interp_gate_out * o_content + (1 - interp_gate_out) * o_prev, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = copy.deepcopy(O_CONTENT[1])
		
	return OUT_BUFFER

def interpolate_dinterp_gate_out(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
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
		_ntm_module2.interpolate_dinterp_gate_out(O_CONTENT[0], O_CONTENT[1], O_PREV[0], OUT_BUFFER[0], gpu_ind)
	else: 
		############ CPU
		interp_gate_out = return_buffer(INTERP_GATE_OUT,gpu_ind)
		o_content = return_buffer(O_CONTENT,gpu_ind)
		o_prev = return_buffer(O_PREV,gpu_ind)
		
		temp = o_content - o_prev
		temp2 = np.zeros((temp.shape[0], temp.shape[1], interp_gate_out.shape[0], 1),dtype='single')
		
		for i in range(temp2.shape[0]):
			for j in range(temp2.shape[1]):
				temp2[i,j,i] = temp[i,j]
				
		OUT_BUFFER = set_buffer(temp2, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((O_CONTENT[1], INTERP_GATE_OUT[1])))
	return OUT_BUFFER
	
def interpolate_do_content(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
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
		_ntm_module2.interpolate_do_content(INTERP_GATE_OUT[0], O_PREV[1], OUT_BUFFER[0], gpu_ind)
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
				
		OUT_BUFFER = set_buffer(temp2, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((O_CONTENT[1], O_CONTENT[1])))
	return OUT_BUFFER
	
def interpolate_do_prev(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
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
		_ntm_module2.interpolate_do_prev(INTERP_GATE_OUT[0], O_PREV[1], OUT_BUFFER[0], gpu_ind)
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
				
		OUT_BUFFER = set_buffer(temp2, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((O_CONTENT[1], O_PREV[1])))
	return OUT_BUFFER

def add_interpolate_layer(LAYERS, name, source):
	assert isinstance(name, str)
	assert isinstance(source, list)
	assert len(source) == 3
	assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
	
	in_shape = [None]*3
	
	for arg in range(3):
		source[arg] = find_layer(LAYERS, source[arg])
		assert source[arg] is not None, 'could not find source layer %i' % arg
		in_shape[arg] = LAYERS[source[arg]]['out_shape']
	
	out_shape = LAYERS[source[1]]['out_shape']
	
	assert len(out_shape) == len(LAYERS[source[1]]['out_shape']) == 2
	assert out_shape == LAYERS[source[1]]['out_shape']
	assert out_shape[0] == LAYERS[source[0]]['out_shape'][0]
	assert LAYERS[source[0]]['out_shape'][1] == 1
	
	LAYERS.append({ 'name': name, 'forward_F': interpolate, \
				'out_shape': LAYERS[source[1]]['out_shape'], \
				'in_shape': in_shape, \
				'in_source': source, \
				'deriv_F': [interpolate_dinterp_gate_out, interpolate_do_content, interpolate_do_prev] })
	
	check_network(LAYERS)
	return len(LAYERS)-1