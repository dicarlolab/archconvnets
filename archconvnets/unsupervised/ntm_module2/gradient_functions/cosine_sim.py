import numpy as np
import archconvnets.unsupervised.ntm_module2._ntm_module2 as _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *

def cosine_sim(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	KEYS, MEM = args
	check_buffer(KEYS)
	check_buffer(MEM)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(KEYS[1]) == len(MEM[1]) == 2
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
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
	assert len(KEYS[1]) == len(MEM[1]) == 2
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim_dmem(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	else:
		########## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
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
		
	OUT_BUFFER[1] = (n_controllers, M, M, mem_length)
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
	assert len(KEYS[1]) == len(MEM[1]) == 2
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim_dkeys(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
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
		
	OUT_BUFFER[1] = (n_controllers, M, n_controllers, mem_length)
	return OUT_BUFFER
