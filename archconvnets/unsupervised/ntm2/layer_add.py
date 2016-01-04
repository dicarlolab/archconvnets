import numpy as np
from gpu_flag import *
from ntm_core import *
from archconvnets.unsupervised.ntm_module2.ntm_gradients import *

def find_layer(LAYERS, name):
	for layer_ind in range(len(LAYERS)):
		if LAYERS[layer_ind]['name'] == name:
			return layer_ind
	return None

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

def add_sum_layer(LAYERS, name, source=None):
	assert isinstance(name, str)
	assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
	
	# default to previous layer as input
	if source is None:
		in_source = len(LAYERS)-1
		in_shape = LAYERS[in_source]['out_shape']
	# find layer specified
	elif isinstance(source,str):
		in_source = find_layer(LAYERS, source)
		assert in_source is not None, 'could not find source layer %i' % source
		in_shape = LAYERS[in_source]['out_shape']
	else:
		assert False, 'unknown source input'
	
	LAYERS.append({ 'name': name, 'forward_F': sum_points, \
				'out_shape': (1,), \
				'in_shape': [in_shape], \
				'in_source': [in_source], \
				'deriv_F': [sum_points_dinput] })
	
	check_network(LAYERS)
	return LAYERS

def add_linear_F_layer(LAYERS, name, n_filters, source=None, random_function=random_function):
	assert isinstance(name, str)
	assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
	
	in_shape = [None]*2
	
	# default to previous layer as input
	if source is None:
		in_source = len(LAYERS)-1
		in_shape[1] = LAYERS[in_source]['out_shape']
	# find layer specified
	elif isinstance(source,str):
		in_source = find_layer(LAYERS, source)
		assert in_source is not None, 'could not find source layer %i' % source
		in_shape[1] = LAYERS[in_source]['out_shape']
	
	# input is user supplied
	elif isinstance(source,tuple):
		in_shape[1] = source
		in_source = -1
	else:
		assert False, 'unknown source input'
	
	in_shape[0] = (n_filters, in_shape[1][0])
	
	LAYERS.append({ 'name': name, 'forward_F': linear_F, \
				'out_shape': (in_shape[0][0], in_shape[1][1]), \
				'in_shape': in_shape, \
				'in_source': [random_function, in_source], \
				'deriv_F': [linear_F_dF, linear_F_dx] })
	
	check_network(LAYERS)
	return LAYERS

if GPU == False:
	def add_focus_keys_layer(LAYERS, name, source):
		assert isinstance(name, str)
		assert isinstance(source, list)
		assert len(source) == 2
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		
		in_shape = [None]*2
		
		source[0] = find_layer(LAYERS, source[0])
		assert source[0] is not None, 'could not find source layer 0'
		
		if source[1] != -1:
			source[1] = find_layer(LAYERS, source[1])
		
		in_shape[0] = LAYERS[source[0]]['out_shape']
		in_shape[1] = (in_shape[0][0], 1)
		
		LAYERS.append({ 'name': name, 'forward_F': focus_keys, \
					'out_shape': LAYERS[source[0]]['out_shape'], \
					'in_shape': in_shape, \
					'in_source': source, \
					'deriv_F': [focus_key_dkeys, focus_key_dbeta_out] })
		
		check_network(LAYERS)
		return LAYERS