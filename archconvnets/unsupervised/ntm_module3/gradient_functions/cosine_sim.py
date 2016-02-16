import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
import time

t_main = [0,0,0]

# keys: N_CONTROLLERS, M_LENGTH
# mem: N_MEM_SLOTS, M_LENGTH
# out: N_CONTROLLERS, N_MEM_SLOTS
# additional_args[0]: batch_imgs (separate correlations batched on first dim)
def cosine_sim(args, OUT_BUFFER=None, additional_args=[False], gpu_ind=GPU_IND):
	t = time.time()
	
	batch_imgs = additional_args[0]
	
	KEYS, MEM = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	if batch_imgs:
		n_imgs = KEYS[1][0]
	else:
		n_imgs = 1
	
	_ntm_module3.cosine_sim(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], n_imgs, gpu_ind)
	
	if batch_imgs:
		OUT_BUFFER[1] = (n_imgs, KEYS[1][1], MEM[1][1])
	else:
		OUT_BUFFER[1] = (KEYS[1][0], MEM[1][0])
	
	if DEBUG:
		check_buffer(KEYS)
		check_buffer(MEM)
		assert isinstance(gpu_ind,int)
		check_buffer(OUT_BUFFER)
		assert (len(MEM[1]) == 2) or ((len(MEM[1]) == 3) and (MEM[1][2] == 1))
		assert (len(KEYS[1]) == 2) or ((len(KEYS[1]) == 3) and (KEYS[1][2] == 1))
		assert KEYS[0] != MEM[0]
		assert OUT_BUFFER[0] != KEYS[0]
		assert OUT_BUFFER[0] != MEM[0]
		assert KEYS[1][1] == MEM[1][1]
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

def cosine_sim_dmem(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	batch_imgs = additional_args[0]
	
	KEYS, MEM = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	if batch_imgs:
		n_imgs = KEYS[1][0]
	else:
		n_imgs = 1
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	DERIV_ABOVE_reshaped = (np.prod(DERIV_ABOVE[1][:n_dim_not_summed]),) + DERIV_ABOVE[1][n_dim_not_summed:]
	
	_ntm_module3.cosine_sim_dmem(KEYS[0], KEYS[1], MEM[0], MEM[1], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], n_imgs, gpu_ind)
	
	OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed] + MEM[1]
	
	if DEBUG:
		check_buffer(OUT_BUFFER_TEMP)
		assert isinstance(gpu_ind,int)
		check_buffer(KEYS)
		check_buffer(MEM)
		check_buffer(LAYER_OUT)
		check_buffer(DERIV_ABOVE)
		check_buffer(OUT_BUFFER)
		assert (len(MEM[1]) == 2) or ((len(MEM[1]) == 3) and (MEM[1][2] == 1))
		assert (len(KEYS[1]) == 2) or ((len(KEYS[1]) == 3) and (KEYS[1][2] == 1))
		assert KEYS[0] != MEM[0]
		assert OUT_BUFFER[0] != KEYS[0]
		assert OUT_BUFFER[0] != MEM[0]
		assert KEYS[1][1] == MEM[1][1]
	
	t_main[1] += time.time() - t
	return OUT_BUFFER

def cosine_sim_dkeys(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	KEYS, MEM = args
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	if batch_imgs:
		n_imgs = X[1][0]
	else:
		n_imgs = 1
	
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	DERIV_ABOVE_reshaped = (np.prod(DERIV_ABOVE[1][:n_dim_not_summed]),) + DERIV_ABOVE[1][n_dim_not_summed:]
	
	_ntm_module3.cosine_sim_dkeys(KEYS[0], KEYS[1], MEM[0], MEM[1], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], n_imgs, gpu_ind)
	
	OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed] + KEYS[1]
	
	if DEBUG:
		check_buffer(OUT_BUFFER_TEMP)
		assert isinstance(gpu_ind,int)
		check_buffer(OUT_BUFFER)
		check_buffer(KEYS)
		check_buffer(MEM)
		check_buffer(LAYER_OUT)
		check_buffer(DERIV_ABOVE)
		assert (len(MEM[1]) == 2) or ((len(MEM[1]) == 3) and (MEM[1][2] == 1))
		assert (len(KEYS[1]) == 2) or ((len(KEYS[1]) == 3) and (KEYS[1][2] == 1))
		assert KEYS[0] != MEM[0]
		assert OUT_BUFFER[0] != KEYS[0]
		assert OUT_BUFFER[0] != MEM[0]
		assert KEYS[1][1] == MEM[1][1]
	
	t_main[2] += time.time() - t
	return OUT_BUFFER

# keys: N_CONTROLLERS, M_LENGTH
# mem: N_MEM_SLOTS, M_LENGTH
def add_cosine_sim_layer(LAYERS, name, source, mem_shape=None, batch_imgs=False, init=0):
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
		if batch_imgs:
			assert (len(in_shape[0]) == 3) or ((len(in_shape[0]) == 4) and (in_shape[0][3] == 1)), 'ndim != 2 + imgs, arg0'
		else:
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
		
		if batch_imgs:
			assert (len(in_shape[1]) == 3) or ((len(in_shape[1]) == 4) and (in_shape[1][3] == 1)), 'ndim != 2 + imgs, arg1'
			assert in_shape[0][2] == in_shape[1][2]
			n_imgs = in_shape[0][0]
			assert n_imgs == in_shape[1][0]
			out_shape = (n_imgs, in_shape[0][1], in_shape[1][1])
		else:
			assert (len(in_shape[1]) == 2) or ((len(in_shape[1]) == 3) and (in_shape[1][2] == 1)), 'ndim != 2, arg1'
			assert in_shape[0][1] == in_shape[1][1]
			out_shape = (in_shape[0][0], in_shape[1][0])
		
		LAYERS[layer_ind]['forward_F'] = cosine_sim
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = in_source
		LAYERS[layer_ind]['deriv_F'] = [cosine_sim_dkeys, cosine_sim_dmem]
		LAYERS[layer_ind]['in_prev'] = [False, in_source_prev]
		LAYERS[layer_ind]['additional_forward_args'] = [batch_imgs]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None], [None]]
		
		return layer_ind
