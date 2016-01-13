import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *
from archconvnets.unsupervised.ntm2.ntm_core import *

# keys: N_CONTROLLERS, M_LENGTH
# mem: N_MEM_SLOTS, M_LENGTH
def cosine_sim(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	KEYS, MEM = args
	check_buffer(KEYS)
	check_buffer(MEM)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert (len(MEM[1]) == 2) or ((len(MEM[1]) == 3) and (MEM[1][2] == 1))
	assert (len(KEYS[1]) == 2) or ((len(KEYS[1]) == 3) and (KEYS[1][2] == 1))
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1][:2]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim(KEYS[0], KEYS[1][:2], MEM[0], MEM[1][:2], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
		if len(KEYS[1]) == 3:
			keys = keys[:,:,0]
		if len(MEM[1]) == 3:
			mem = mem[:,:,0]
		numer = np.dot(keys, mem.T)
		denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
		OUT_BUFFER = set_buffer(numer/denom, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = (n_controllers, M)
	return OUT_BUFFER

def cosine_sim_dmem(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	KEYS, MEM = args
	check_buffer(KEYS)
	check_buffer(MEM)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert (len(MEM[1]) == 2) or ((len(MEM[1]) == 3) and (MEM[1][2] == 1))
	assert (len(KEYS[1]) == 2) or ((len(KEYS[1]) == 3) and (KEYS[1][2] == 1))
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1][:2]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim_dmem(KEYS[0], KEYS[1][:2], MEM[0], MEM[1][:2], OUT_BUFFER[0], gpu_ind)
	else:
		########## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
		if len(KEYS[1]) == 3:
			keys = keys[:,:,0]
		if len(MEM[1]) == 3:
			mem = mem[:,:,0]
		comb = np.zeros((n_controllers, mem.shape[0], mem.shape[0], mem.shape[1]),dtype='single')

		keys_sq_sum = np.sqrt(np.sum(keys**2, 1))
		mem_sq_sum = np.sqrt(np.sum(mem**2, 1))

		denom = np.einsum(keys_sq_sum, [0], mem_sq_sum, [1], [0,1])
		numer = np.dot(keys, mem.T)

		numer = numer / denom**2
		denom = 1 / denom # = denom/denom**2

		mem = mem / mem_sq_sum[:,np.newaxis]
		temp = np.einsum(mem, [0,2], numer*keys_sq_sum[:,np.newaxis], [1,0], [1,0,2])
		keys_denom = keys[:,np.newaxis] * denom[:,:,np.newaxis]
		
		comb[:,range(mem.shape[0]),range(mem.shape[0])] = keys_denom - temp
		OUT_BUFFER = set_buffer(comb, OUT_BUFFER, gpu_ind)
	
	if len(MEM[1]) == 2:
		OUT_BUFFER[1] = (n_controllers, M, M, mem_length)
	else:
		OUT_BUFFER[1] = (n_controllers, M, M, mem_length,1)
	return OUT_BUFFER

def cosine_sim_dkeys(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	KEYS, MEM = args
	check_buffer(KEYS)
	check_buffer(MEM)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert (len(MEM[1]) == 2) or ((len(MEM[1]) == 3) and (MEM[1][2] == 1))
	assert (len(KEYS[1]) == 2) or ((len(KEYS[1]) == 3) and (KEYS[1][2] == 1))
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1][:2]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim_dkeys(KEYS[0], KEYS[1][:2], MEM[0], MEM[1][:2], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
		if len(KEYS[1]) == 3:
			keys = keys[:,:,0]
		if len(MEM[1]) == 3:
			mem = mem[:,:,0]
		comb = np.zeros((n_controllers, mem.shape[0], n_controllers, keys.shape[1]),dtype='single')
		
		keys_sq_sum = np.sqrt(np.sum(keys**2, 1))
		mem_sq_sum = np.sqrt(np.sum(mem**2, 1))
		
		denom = np.einsum(keys_sq_sum, [0], mem_sq_sum, [1], [0,1])
		numer = np.dot(keys, mem.T)
		
		numer = numer / denom**2
		denom = 1 / denom # = denom/denom**2
		
		keys = keys / keys_sq_sum[:,np.newaxis]
		temp = np.einsum(keys, [1,2], numer*mem_sq_sum[np.newaxis], [1,0], [1,0,2])
		mem_denom = mem[np.newaxis] * denom[:,:,np.newaxis]
		
		comb[range(n_controllers),:,range(n_controllers)] = mem_denom - temp
		OUT_BUFFER = set_buffer(comb, OUT_BUFFER, gpu_ind)
		
	if len(KEYS[1]) == 2:
		OUT_BUFFER[1] = (n_controllers, M, n_controllers, mem_length)
	else:
		OUT_BUFFER[1] = (n_controllers, M, n_controllers, mem_length, 1)
	return OUT_BUFFER

# keys: N_CONTROLLERS, M_LENGTH
# mem: N_MEM_SLOTS, M_LENGTH
def add_cosine_sim_layer(LAYERS, name, source, mem_shape=None, init=0):
	assert isinstance(name, str)
	assert len(source) == 2
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		in_shape = [None]*2
		in_source = [None]*2
		
		# arg 0
		in_source[0] = find_layer(LAYERS, source[0])
		assert in_source[0] is not None, 'could not find source layer %i' % source[0]
		in_shape[0] = LAYERS[in_source[0]]['out_shape']
		assert (len(in_shape[0]) == 2) or ((len(in_shape[0]) == 3) and (in_shape[0][2] == 1)), 'ndim != 2, arg0'
		
		# arg 1
		in_source[1] = find_layer(LAYERS, source[1])
		assert in_source[1] is not None, 'could not find source layer %i' % source[1]
		if source[1][-1] != '-': # input is another layer
			in_source_prev = False
			in_shape[1] = LAYERS[in_source[1]]['out_shape']
		else:
			in_source_prev = True
			assert mem_shape is not None, 'shape must be provided if layer is a memory layer'
			in_shape[1] = mem_shape
		assert (len(in_shape[1]) == 2) or ((len(in_shape[1]) == 3) and (in_shape[1][2] == 1)), 'ndim != 2, arg1'
		assert in_shape[0][1] == in_shape[1][1]
		
		out_shape = (in_shape[0][0], in_shape[1][0])
		
		LAYERS[layer_ind]['forward_F'] = cosine_sim
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = in_source
		LAYERS[layer_ind]['deriv_F'] = [cosine_sim_dkeys, cosine_sim_dmem]
		LAYERS[layer_ind]['in_prev'] = [False, in_source_prev]
		
		return layer_ind
