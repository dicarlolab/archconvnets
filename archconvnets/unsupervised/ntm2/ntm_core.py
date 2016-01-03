import numpy as np
deriv_top = np.ones((1,1))

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
				assert WEIGHTS[layer_ind][arg].shape == L['in_shape'][arg], 'layer %i argument %i not initialized to right size' % (layer_ind, arg)
			else:
				assert WEIGHTS[layer_ind][arg] is None, 'layer %i argument %i should not have weightings because it should be computed from layer %i' % (layer_ind, arg,  L['in_source'][arg])
		
def check_network(LAYERS):
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		assert len(L['in_shape']) == len(L['deriv_F']) == len(L['in_source'])
		
		# build arguments
		N_ARGS = len(L['in_shape'])
		args = [None] * N_ARGS
		for arg in range(N_ARGS):
			args[arg] = np.asarray(np.random.random(L['in_shape'][arg]),dtype='single')
		
		# check if function corretly produces specified output dimensions
		layer_output = L['forward_F'](args)
		assert layer_output.shape == L['out_shape'], "%i" % (layer_ind)
		
		# check if deriv functions correctly produce correct shapes
		for arg in range(N_ARGS):
			expected_shape = tuple(np.concatenate((L['out_shape'], L['in_shape'][arg])))
			assert L['deriv_F'][arg](args, layer_output).shape == expected_shape
		
		# check if other layers claim to produce expected inputs
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_shape'][arg] == LAYERS[L['in_source'][arg]]['out_shape'], '%i %i' % (layer_ind, arg)
				
		# check if layers are ordered (no inputs to this layer come after this one in the list... unless recursive mem layer)
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_source'][arg] <= layer_ind or layer_ind in LAYERS[L['in_source'][arg]]['in_source']

def check_output_prev(OUTPUT_PREV, LAYERS):
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		if layer_ind in L['in_source']:
			assert OUTPUT_PREV[layer_ind].shape == L['out_shape']
			
def init_weights(LAYERS):
	check_network(LAYERS)
	WEIGHTS = [None]*len(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_INPUTS = len(L['in_shape'])
		WEIGHTS[layer_ind] = [None]*N_INPUTS
		for arg in range(N_INPUTS):
			if isinstance(L['in_source'][arg], int) != True:
				WEIGHTS[layer_ind][arg] = L['in_source'][arg]( L['in_shape'][arg] )
				
	return WEIGHTS

def mult_partials(A, B, B_out_shape):
	A_ndim = A.ndim - len(B_out_shape)
	B_ndim = B.ndim - len(B_out_shape)
	assert A_ndim > 0
	assert B_ndim > 0
	assert A.shape[A_ndim:] == B.shape[:len(B_out_shape)]
	
	A_dim0 = np.prod(A.shape[:A_ndim])
	B_dim1 = np.prod(B.shape[len(B_out_shape):])
	collapsed = np.prod(B_out_shape)

	A_shape = (A_dim0, collapsed)
	B_shape = (collapsed, B_dim1)
	
	out_shape = np.concatenate((A.shape[:A_ndim], B.shape[len(B_out_shape):]))
	
	return np.dot(A.reshape(A_shape), B.reshape(B_shape)).reshape(out_shape)

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

def forward_network(LAYERS, WEIGHTS, OUTPUT_PREV):
	OUTPUT = [None] * len(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_shape'])
		args = build_forward_args(L, layer_ind, OUTPUT, OUTPUT_PREV, WEIGHTS)
		
		OUTPUT[layer_ind] = L['forward_F'](args)
	return OUTPUT

def local_derivs(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV):
	LOCAL_DERIVS = [None] * len(LAYERS)
	
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_shape'])
		LOCAL_DERIVS[layer_ind] = [None]*N_ARGS
		
		args = build_forward_args(L, layer_ind, OUTPUT, OUTPUT_PREV, WEIGHTS)
		
		for arg in range(N_ARGS):
			LOCAL_DERIVS[layer_ind][arg] = L['deriv_F'][arg](args, OUTPUT[layer_ind])
	return LOCAL_DERIVS

def add_initialize(W, add):
	if W is None:
		return add
	else:
		return W + add

def reverse_network(deriv_above, layer_ind, LAYERS, LOCAL_DERIVS, PARTIALS, WEIGHT_DERIVS=None, keep_dims=False): # multiply all partials together
	if WEIGHT_DERIVS is None:
		WEIGHT_DERIVS = [None] * len(LAYERS)
		for layer_ind in range(len(LAYERS)):
			WEIGHT_DERIVS[layer_ind] = [None]*len(LAYERS[layer_ind]['in_shape'])

	return reverse_network_recur(deriv_above, layer_ind, LAYERS, LOCAL_DERIVS, PARTIALS, WEIGHT_DERIVS, keep_dims)
	
def reverse_network_recur(deriv_above, layer_ind, LAYERS, LOCAL_DERIVS, PARTIALS, WEIGHT_DERIVS, keep_dims): # multiply all partials together
	L = LAYERS[layer_ind]
	P = PARTIALS[layer_ind]
	N_ARGS = len(L['in_shape'])
	
	for arg in range(N_ARGS):
		deriv_above_new = mult_partials(deriv_above, LOCAL_DERIVS[layer_ind][arg], LAYERS[layer_ind]['out_shape'])
		src = LAYERS[layer_ind]['in_source'][arg]
		
		# go back farther... avoid infinite loops
		if isinstance(src, int) and src != -1 and src != layer_ind:
			reverse_network_recur(deriv_above_new, src, LAYERS, LOCAL_DERIVS, PARTIALS, WEIGHT_DERIVS, keep_dims)
		
		# add memory partials
		elif src == layer_ind:
			N_ARGS2 = len(P['in_source'])
			for arg2 in range(N_ARGS2):
				p_layer_ind = P['in_source'][arg2]
				p_arg = P['in_arg'][arg2]
				p_partial = P['partial'][arg2]
				deriv_temp = mult_partials(deriv_above_new, p_partial, LAYERS[layer_ind]['out_shape'])
				WEIGHT_DERIVS[p_layer_ind][p_arg] = add_initialize(WEIGHT_DERIVS[p_layer_ind][p_arg], deriv_temp)
		
		# regular derivatives
		elif keep_dims:
			WEIGHT_DERIVS[layer_ind][arg] = add_initialize(WEIGHT_DERIVS[layer_ind][arg], deriv_above_new)
		else:  # remove first singleton dimension
			assert deriv_above_new.shape[0] == 1
			WEIGHT_DERIVS[layer_ind][arg] = add_initialize(WEIGHT_DERIVS[layer_ind][arg], deriv_above_new[0])
	return WEIGHT_DERIVS

def init_traverse_to_end(layer_orig, layer_cur, arg, LAYERS, PARTIALS):
	dest = LAYERS[layer_cur]['in_source'][arg]
	# end:
	if (isinstance(dest, int) == False) or dest == -1:
		PARTIALS[layer_orig]['in_source'].append(layer_cur)
		PARTIALS[layer_orig]['in_arg'].append(arg)
		PARTIALS[layer_orig]['partial'].append(np.zeros(np.concatenate((LAYERS[layer_orig]['out_shape'], LAYERS[layer_cur]['in_shape'][arg])), dtype='single'))
	else:
		N_ARGS2 = len(LAYERS[dest]['in_source'])
		for arg2 in range(N_ARGS2):
			init_traverse_to_end(layer_orig, dest, arg2, LAYERS, PARTIALS)
	
def init_partials(LAYERS):
	PARTIALS = [None]*len(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_source'])
		PARTIALS[layer_ind] = {'in_source': [], 'in_arg': [], 'partial': []}
		
		if layer_ind in L['in_source']: # memory layer
			for arg in range(N_ARGS):
				if L['in_source'][arg] != layer_ind:
					init_traverse_to_end(layer_ind, layer_ind, arg, LAYERS, PARTIALS)
		
	return PARTIALS

def copy_traverse_to_end(layer_orig, layer_cur, arg, LAYERS, PARTIALS, MEM_WEIGHT_DERIVS):
	dest = LAYERS[layer_cur]['in_source'][arg]
	# end:
	if (isinstance(dest, int) == False) or dest == -1:
		PARTIALS[layer_orig]['partial'].append(MEM_WEIGHT_DERIVS[layer_cur][arg])
	else:
		N_ARGS2 = len(LAYERS[dest]['in_source'])
		for arg2 in range(N_ARGS2):
			copy_traverse_to_end(layer_orig, dest, arg2, LAYERS, PARTIALS, MEM_WEIGHT_DERIVS)

def copy_partials(layer_ind, LAYERS, PARTIALS_PREV, MEM_WEIGHT_DERIVS):
	L = LAYERS[layer_ind]
	N_ARGS = len(L['in_source'])
	
	for arg in range(N_ARGS):
		if L['in_source'][arg] != layer_ind:
			PARTIALS_PREV[layer_ind]['partial'] = []
			copy_traverse_to_end(layer_ind, layer_ind, arg, LAYERS, PARTIALS_PREV, MEM_WEIGHT_DERIVS)
	return PARTIALS_PREV

def reverse_mem_network(MEM_IND, LAYERS, LOCAL_DERIVS, PARTIALS_PREV):
	MEM_WEIGHT_DERIVS = [None] * len(LAYERS)
	for layer_ind in range(len(LAYERS)):
		MEM_WEIGHT_DERIVS[layer_ind] = [None]*len(LAYERS[layer_ind]['in_shape'])
			
	# update partials_prev
	for i in range(len(LAYERS[MEM_IND]['in_source'])):
		src_layer = LAYERS[MEM_IND]['in_source'][i]
		
		# DMEM_D[layers]_DW
		if src_layer != MEM_IND:
			reverse_network(LOCAL_DERIVS[MEM_IND][i], src_layer, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, MEM_WEIGHT_DERIVS, keep_dims=True)
		
		# DMEM_DMEM_DW
		else:
			P = PARTIALS_PREV[MEM_IND]
			N_ARGS2 = len(P['in_source'])
			for arg2 in range(N_ARGS2):
				p_layer_ind = P['in_source'][arg2]
				p_arg = P['in_arg'][arg2]
				p_partial = P['partial'][arg2]
				deriv_temp = mult_partials(LOCAL_DERIVS[MEM_IND][i], p_partial, LAYERS[MEM_IND]['out_shape'])
				MEM_WEIGHT_DERIVS[p_layer_ind][p_arg] = add_initialize(MEM_WEIGHT_DERIVS[p_layer_ind][p_arg], deriv_temp)
				
				
	return MEM_WEIGHT_DERIVS