import numpy as np
from ntm_core import *

def find_layer(LAYERS, name):
	for layer_ind in range(len(LAYERS)):
		if LAYERS[layer_ind]['name'] == name:
			return layer_ind
	return None

def cosine_sim_dmem(args, layer_out):
	assert len(args) == 2
	keys, mem = args
	n_controllers = keys.shape[0]
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
	return comb

def cosine_sim_dkeys(args, layer_out):
	assert len(args) == 2
	keys, mem = args
	n_controllers = keys.shape[0]
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
	return comb

def cosine_sim(args):
	assert len(args) == 2
	keys, mem = args
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	return numer / denom # [n_controllers, n_mem_slots]

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

def add_points(args):
	assert len(args) == 2
	assert args[0].shape == args[1].shape
	return args[0] + args[1]

def add_points_dinput(args, layer_out):
	assert len(args) == 2
	assert args[0].shape == args[1].shape
	out = np.zeros(np.concatenate((args[0].shape, args[0].shape)),dtype='single')
	for i in range(out.shape[0]):
		out[i,range(out.shape[1]),i,range(out.shape[1])] = 1
	return out

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


def sum_points(args):
	assert len(args) == 1
	return args[0].sum()[np.newaxis]

def sum_points_dinput(args, layer_out):
	assert len(args) == 1
	return np.ones(tuple(np.concatenate(((1,), args[0].shape))))

def sq_points(args):
	assert len(args) == 1
	input = args[0]
	return input**2

def sq_points_dinput(args, layer_out):
	input = args[0]
	n = input.shape[1]
	dinput = np.zeros((input.shape[0], n, input.shape[0], n),dtype='single')
	for i in range(input.shape[0]):
		dinput[i,range(n),i,range(n)] = 2*input[i]
	return dinput

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

def linear_F(args):
	F, layer_in = args
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in) # [n1, 1]

def linear_F_dx(args, layer_out):
	F, x = args
	n = x.shape[1]
	temp = np.zeros((F.shape[0], n, x.shape[0], n),dtype='single')
	temp[:,range(n),:,range(n)] = F
	return temp

def linear_F_dF(args, layer_out):
	F, x = args
	n = F.shape[0]
	temp = np.zeros((n, x.shape[1], n, F.shape[1]),dtype='single')
	temp[range(n),:,range(n)] = x.T
	return temp

############
##### check!
# softmax over second dimension; first dim. treated independently
def softmax(args):
	assert len(args) == 1
	layer_in = args[0]
	exp_layer_in = np.exp(layer_in)
	return exp_layer_in/np.sum(exp_layer_in,1)[:,np.newaxis]

def softmax_dlayer_in(args, layer_out):
	assert len(args) == 1
	g = np.zeros((layer_out.shape[0], layer_out.shape[1], layer_out.shape[0], layer_out.shape[1]),dtype='single')
	
	# dsoftmax[:,i]/dlayer_in[:,j] when i = j:
	temp = (layer_out * (1 - layer_out))
	for i in range(g.shape[0]):
		for j in range(g.shape[1]):
			g[i,j,i,j] = temp[i,j]
	
	# i != j
	for i in range(g.shape[0]):
		for j in range(g.shape[1]):
			for k in range(g.shape[1]):
				if j != k:
					g[i,j,i,k] -= layer_out[i,j]*layer_out[i,k]
	return g

##########
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

# focus keys, scalar beta_out (one for each controller) multiplied with each of its keys
def focus_keys(args):
	keys, beta_out = args
	# keys: [n_controllers, m_length], beta_out: [n_controllers, 1]
	
	return keys * beta_out # [n_controllers, m_length]

def focus_key_dbeta_out(args, layer_out): 
	keys, beta_out = args
	# beta_out: [n_controllers, 1]
	n_controllers, m_length = keys.shape
	
	g = np.zeros((n_controllers, m_length, n_controllers, 1),dtype='single')
	g[range(n_controllers),:,range(n_controllers),0] = keys
	return g

def focus_key_dkeys(args, layer_out): 
	keys, beta_out = args
	# beta_out: [n_controllers, 1]
	n_controllers, m_length = keys.shape
	
	g = np.zeros((n_controllers, m_length, n_controllers, m_length),dtype='single')
	for j in range(m_length):
		g[range(n_controllers),j,range(n_controllers),j] = np.squeeze(beta_out)
	return g
