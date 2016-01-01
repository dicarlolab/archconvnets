import numpy as np
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *

def random_function(size):
	return np.asarray(np.random.random(size) - .5, dtype='single')

def check_weights(WEIGHTS, LAYERS):
	check_network(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_shape'])
		
		for arg in range(N_ARGS):
			if isinstance(L['in_source'][arg], int) == False or L['in_source'][arg] == -1:
				assert WEIGHTS[layer_ind][arg] is not None, 'layer %i argument %i not initialized' % (layer_ind, arg)
				assert WEIGHTS[layer_ind][arg][1] == L['in_shape'][arg], 'layer %i argument %i not initialized to right size' % (layer_ind, arg)
			else:
				assert WEIGHTS[layer_ind][arg] is None, 'layer %i argument %i should not have weightings because it should be computed from layer %i' % (layer_ind, arg,  L['in_source'][arg])


def check_network(LAYERS):
	n_allocated = return_n_allocated()
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		assert len(L['in_shape']) == len(L['deriv_F']) == len(L['in_source'])
		
		# build arguments
		N_ARGS = len(L['in_shape'])
		args = [None] * N_ARGS
		for arg in range(N_ARGS):
			args[arg] = init_buffer(np.asarray(np.random.random(L['in_shape'][arg]),dtype='single'))
		
		# check if function corretly produces specified output dimensions
		OUT = L['forward_F'](args)
		assert OUT[1] == L['out_shape'], "%i" % (layer_ind)
		free_buffer(OUT)
		
		# check if deriv functions correctly produce correct shapes
		for arg in range(N_ARGS):
			expected_shape = tuple(np.concatenate((L['out_shape'], L['in_shape'][arg])))
			OUT = L['deriv_F'][arg](args)
			assert OUT[1] == expected_shape
			free_buffer(OUT)
		
		# free mem
		for arg in range(N_ARGS):
			free_buffer(args[arg])
		
		# check if other layers claim to produce expected inputs
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_shape'][arg] == LAYERS[L['in_source'][arg]]['out_shape'], '%i %i' % (layer_ind, arg)
				
		# check if layers are ordered (no inputs to this layer come after this one in the list... unless recursive mem layer)
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_source'][arg] <= layer_ind or layer_ind in LAYERS[L['in_source'][arg]]['in_source']
	assert n_allocated == return_n_allocated(), 'check_network() leaked memory'

def check_output_prev(OUTPUT_PREV, LAYERS):
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		if layer_ind in L['in_source']:
			assert OUTPUT_PREV[layer_ind][1] == L['out_shape']

def init_weights(LAYERS):
	check_network(LAYERS)
	WEIGHTS = [None]*len(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_INPUTS = len(L['in_shape'])
		WEIGHTS[layer_ind] = [None]*N_INPUTS
		for arg in range(N_INPUTS):
			if isinstance(L['in_source'][arg], int) != True:
				WEIGHTS[layer_ind][arg] = init_buffer(L['in_source'][arg]( L['in_shape'][arg] ))
			elif L['in_source'][arg] == -1: # user supplied
				WEIGHTS[layer_ind][arg] = init_buffer()
				
	return WEIGHTS

def build_forward_args(L, layer_ind, OUTPUT, OUTPUT_PREV, WEIGHTS):
	N_ARGS = len(L['in_shape'])
	args = [None] * N_ARGS
	
	for arg in range(N_ARGS):
		src = L['in_source'][arg]
		
		# input is from another layer
		if isinstance(src, int) and src != -1 and src != layer_ind:
			args[arg] = OUTPUT[src]
		# input is current layer, return previous value
		elif src == layer_ind:
			args[arg] = OUTPUT_PREV[src]
		else: # input is a weighting
			args[arg] = WEIGHTS[layer_ind][arg]
	return args

def forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV):
	assert len(OUTPUT_PREV) == len(LAYERS)
	if OUTPUT is None:
		OUTPUT = [None] * len(LAYERS)
		for layer_ind in range(len(LAYERS)):
			OUTPUT[layer_ind] = init_buffer()
	
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_shape'])
		args = build_forward_args(L, layer_ind, OUTPUT, OUTPUT_PREV, WEIGHTS)
		
		L['forward_F'](args, OUTPUT[layer_ind])
	return OUTPUT
	
	