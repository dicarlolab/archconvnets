import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *

##########
# focus keys, scalar beta_out (one for each controller) multiplied with each of its keys
def focus_keys(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	KEYS, BETA_OUT = args
	check_buffer(KEYS)
	check_buffer(BETA_OUT)
	assert KEYS[1][0] == BETA_OUT[1][0]
	assert len(KEYS[1]) == len(BETA_OUT[1]) == 2
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module3.focus_key(KEYS[0], KEYS[1], BETA_OUT[0], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS,gpu_ind)
		beta_out = return_buffer(BETA_OUT,gpu_ind)
		# keys: [n_controllers, m_length], beta_out: [n_controllers, 1]
		
		OUT_BUFFER = set_buffer(keys * beta_out, OUT_BUFFER, gpu_ind) # [n_controllers, m_length]
		
	OUT_BUFFER[1] = copy.deepcopy(KEYS[1])
	return OUT_BUFFER

def focus_key_dbeta_out(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	KEYS, BETA_OUT = args
	check_buffer(KEYS)
	check_buffer(BETA_OUT)
	check_buffer(DERIV_ABOVE)
	assert KEYS[1][0] == BETA_OUT[1][0]
	assert len(KEYS[1]) == len(BETA_OUT[1]) == 2
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	n_controllers, m_length = KEYS[1]
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.focus_key_dbeta_out(KEYS[0], KEYS[1], OUT_BUFFER_TEMP[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS,gpu_ind)
		beta_out = return_buffer(BETA_OUT,gpu_ind)
		# beta_out: [n_controllers, 1]
		
		g = np.zeros((n_controllers, m_length, n_controllers, 1),dtype='single')
		g[range(n_controllers),:,range(n_controllers),0] = keys
		OUT_BUFFER_TEMP = set_buffer(g, OUT_BUFFER_TEMP, gpu_ind)
		
	OUT_BUFFER_TEMP[1] = (n_controllers, m_length, n_controllers, 1)
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	
	return OUT_BUFFER

def focus_key_dkeys(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert additional_args == [None]
	KEYS, BETA_OUT = args
	check_buffer(KEYS)
	check_buffer(BETA_OUT)
	check_buffer(DERIV_ABOVE)
	assert KEYS[1][0] == BETA_OUT[1][0]
	assert len(KEYS[1]) == len(BETA_OUT[1]) == 2
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	n_controllers, m_length = KEYS[1]
	
	OUT_BUFFER_TEMP = init_buffer(gpu_ind=gpu_ind)
	
	if GPU:
		_ntm_module3.focus_key_dkeys(BETA_OUT[0], KEYS[1], OUT_BUFFER_TEMP[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS,gpu_ind)
		beta_out = return_buffer(BETA_OUT,gpu_ind)
		# beta_out: [n_controllers, 1]
		
		g = np.zeros((n_controllers, m_length, n_controllers, m_length),dtype='single')
		for j in range(m_length):
			g[range(n_controllers),j,range(n_controllers),j] = np.squeeze(beta_out)
		OUT_BUFFER_TEMP = set_buffer(g, OUT_BUFFER_TEMP, gpu_ind)
	
	OUT_BUFFER_TEMP[1] = (n_controllers, m_length, n_controllers, m_length)
	check_buffer(OUT_BUFFER_TEMP)
	
	OUT_BUFFER = mult_partials(DERIV_ABOVE, OUT_BUFFER_TEMP, LAYER_OUT[1], OUT_BUFFER)
	free_buffer(OUT_BUFFER_TEMP)
	
	return OUT_BUFFER

def add_focus_keys_layer(LAYERS, name, source, init=0):
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
		
		in_shape = [None]*2
		
		source[0] = find_layer(LAYERS, source[0])
		assert source[0] is not None, 'could not find source layer 0'
		
		source[1] = find_layer(LAYERS, source[1])
		assert source[1] is not None, 'could not find source layer 1'
		
		in_shape[0] = LAYERS[source[0]]['out_shape']
		in_shape[1] = (in_shape[0][0], 1)
		
		LAYERS[layer_ind]['forward_F'] = focus_keys
		LAYERS[layer_ind]['out_shape'] = LAYERS[source[0]]['out_shape']
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = source
		LAYERS[layer_ind]['deriv_F'] = [focus_key_dkeys, focus_key_dbeta_out]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None], [None]]
		
		return layer_ind