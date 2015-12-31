import numpy as np
import copy
import time
import scipy.optimize

N_CONTROLLERS = 16
M_LENGTH = 6
N_MEM_SLOTS = 8
mem_shape = (N_MEM_SLOTS, M_LENGTH)

'''-------------
layer Ul:
	inputs: x[t], U
	outputs: U_OUT[t]

layer Wl:
	inputs: U_OUT[t-1], mem[t-1], W
	outputs: mem[t]

layer Rl:
	inputs: U_OUT[t], mem[t], R
	outputs: O[t]

------------
dOl[t] -> dOl[t]/dR, dO[t]/dU_OUT[t], dO[t]/dmem[t]
	                       dU_OUT[t] -> dU_OUT[t]/dU
	                                   dmem[t] -> dmem[t]/dW, dmem[t]/dU_OUT[t-1], dmem[t]/dmem[t-1]
									                                               dmem[t-1] -> dmem[t-1]/dW, dmem[t-1]/dU_OUT[t-2], dmem[t-1]/dmem[t-2]
																				   dmem[t-2] -> dmem[t-2]/dW, dmem[t-2]/dU_OUT[t-3], dmem[t-2]/dmem[t-3]
																				   dmem[t-3] -> dmem[t-3]/dW, dmem[t-3]/dU_OUT[t-4], 0
																	               dmem[t-4] -> 0
'''								                      

def mem_prev(args):
	return np.zeros(mem_shape, dtype='single')

def mem_prev_deriv(ARGS):
	arg_ind = ARGS[0]
	args = ARGS[1]
	return np.zeros(np.concatenate((mem_shape, args[arg_ind].shape)), dtype='single')

def cosine_sim_dmem(args):
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

def cosine_sim_dkeys(args):
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

def add_points(args):
	assert len(args) == 2
	assert args[0].shape == args[1].shape
	return args[0] + args[1]

def add_points_dinput(args):
	assert len(args) == 2
	assert args[0].shape == args[1].shape
	out = np.zeros(np.concatenate((args[0].shape, args[0].shape)),dtype='single')
	for i in range(out.shape[0]):
		out[i,range(out.shape[1]),i,range(out.shape[1])] = 1
	return out

def sum_points(args):
	assert len(args) == 1
	return args[0].sum()[np.newaxis]

def sum_points_dinput(args):
	assert len(args) == 1
	return np.ones(tuple(np.concatenate(((1,), args[0].shape))))

def sq_points(args):
	assert len(args) == 1
	input = args[0]
	return input**2

def sq_points_dinput(args):
	input = args[0]
	n = input.shape[1]
	dinput = np.zeros((input.shape[0], n, input.shape[0], n),dtype='single')
	for i in range(input.shape[0]):
		dinput[i,range(n),i,range(n)] = 2*input[i]
	return dinput

def linear_F(args):
	F, layer_in = args
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in) # [n1, 1]

def linear_F_dx(args):
	F, x = args
	n = x.shape[1]
	temp = np.zeros((F.shape[0], n, x.shape[0], n),dtype='single')
	temp[:,range(n),:,range(n)] = F
	return temp

def linear_F_dF(args):
	F, x = args
	n = F.shape[0]
	temp = np.zeros((n, x.shape[1], n, F.shape[1]),dtype='single')
	temp[range(n),:,range(n)] = x.T
	return temp

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
		
def call_deriv(L, arg, args):
	if 'deriv_F_args' in L:
		return L['deriv_F'][arg]([L['deriv_F_args'][arg], args])
	else:
		return L['deriv_F'][arg](args)

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
		assert L['forward_F'](args).shape == L['out_shape'], "%i" % (layer_ind)
		
		# check if deriv functions correctly produce correct shapes
		for arg in range(N_ARGS):
			expected_shape = tuple(np.concatenate((L['out_shape'], L['in_shape'][arg])))
			assert call_deriv(L, arg, args).shape == expected_shape
		
		# check if other layers claim to produce expected inputs
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_shape'][arg] == LAYERS[L['in_source'][arg]]['out_shape'], '%i %i' % (layer_ind, arg)
				
		# check if layers are ordered (no inputs to this layer come after this one in the list)
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_source'][arg] <= layer_ind

def init_weights(LAYERS):
	check_network(LAYERS)
	WEIGHTS = [None]*len(LAYERS)
	for i in range(len(LAYERS)):
		L = LAYERS[i]
		N_INPUTS = len(L['in_shape'])
		WEIGHTS[i] = [None]*N_INPUTS
		for j in range(N_INPUTS):
			if isinstance(L['in_source'][j], int) != True:
				WEIGHTS[i][j] = L['in_source'][j](L['in_shape'][j])
				
	return WEIGHTS

def mult_partials(A, layer_ind, arg, LAYERS, DERIVS):
	B = DERIVS[layer_ind][arg]
	B_out_shape = LAYERS[layer_ind]['out_shape']
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

def build_forward_args(L, layer_ind, OUTPUT, WEIGHTS):
	N_ARGS = len(L['in_shape'])
	args = [None] * N_ARGS
	
	for arg in range(N_ARGS):
		# input is from another layer
		if isinstance(L['in_source'][arg], int) and L['in_source'][arg] != -1:
			args[arg] = OUTPUT[L['in_source'][arg]]
		else: # input is a weighting
			args[arg] = WEIGHTS[layer_ind][arg]
	return args

def forward_network(LAYERS, WEIGHTS):
	OUTPUT = [None] * len(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_shape'])
		args = build_forward_args(L, layer_ind, OUTPUT, WEIGHTS)
		
		OUTPUT[layer_ind] = L['forward_F'](args)
	return OUTPUT

def local_derivs(LAYERS, WEIGHTS, OUTPUT):
	DERIVS = [None] * len(LAYERS)
	WEIGHT_DERIVS = [None] * len(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_shape'])
		DERIVS[layer_ind] = [None]*N_ARGS
		WEIGHT_DERIVS[layer_ind] = [None]*N_ARGS
		
		args = build_forward_args(L, layer_ind, OUTPUT, WEIGHTS)
		
		for arg in range(N_ARGS):
			DERIVS[layer_ind][arg] = call_deriv(L, arg, args)
	return DERIVS, WEIGHT_DERIVS

def reverse_network(deriv_above, layer_ind, LAYERS, DERIVS, WEIGHT_DERIVS):
	assert layer_ind >= 0
	L = LAYERS[layer_ind]
	N_ARGS = len(L['in_shape'])
	
	for arg in range(N_ARGS):
		deriv_above_new = mult_partials(deriv_above, layer_ind, arg, LAYERS, DERIVS)
		src = LAYERS[layer_ind]['in_source'][arg]
		if isinstance(src, int) and src != -1 and src != layer_ind: # go back farther
			reverse_network(deriv_above_new, src, LAYERS, DERIVS, WEIGHT_DERIVS)
		else:
			if WEIGHT_DERIVS[layer_ind][arg] is None:
				WEIGHT_DERIVS[layer_ind][arg] = deriv_above_new[0]
			else:
				WEIGHT_DERIVS[layer_ind][arg] += deriv_above_new[0]

#############
deriv_top = np.ones((1,1))
LAYERS = []

# FR read
FR_IND = len(LAYERS)
F1_shape = (N_CONTROLLERS,4)
x_shape = (4,M_LENGTH)
L1_shape = (F1_shape[0], x_shape[1])
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L1_shape, \
				'in_shape': [F1_shape, x_shape], \
				'in_source': [random_function, -1], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# Fw write
FW_IND = len(LAYERS)
Fw_shape = (N_MEM_SLOTS,8)
m_shape = (8,M_LENGTH)
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': mem_shape, \
				'in_shape': [Fw_shape, m_shape], \
				'in_source': [random_function, -1], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# mem_prev
MEM_PREV_IND = len(LAYERS)
LAYERS.append({ 'forward_F': mem_prev, \
				'out_shape': mem_shape})
				
# mem = mem_prev + Fw
# "mem"
MEM_IND = len(LAYERS)
LAYERS.append({ 'forward_F': add_points, \
				'out_shape': mem_shape, \
				'in_shape': [mem_shape, mem_shape], \
				'in_source': [FW_IND, MEM_PREV_IND], \
				'deriv_F': [add_points_dinput, add_points_dinput] })

N_ARGS = len(LAYERS[MEM_IND]['in_shape'])
LAYERS[MEM_PREV_IND]['in_shape'] = LAYERS[MEM_IND]['in_shape']
LAYERS[MEM_PREV_IND]['in_source'] = LAYERS[MEM_IND]['in_source']
LAYERS[MEM_PREV_IND]['deriv_F'] = [mem_prev_deriv] * N_ARGS
LAYERS[MEM_PREV_IND]['deriv_F_args'] = range(N_ARGS)

# cosine
COS_IND = len(LAYERS)
mem_shape = (N_MEM_SLOTS, M_LENGTH)
L2_shape = (N_CONTROLLERS, N_MEM_SLOTS)
LAYERS.append({ 'forward_F': cosine_sim, \
				'out_shape': L2_shape, \
				'in_shape': [L1_shape, mem_shape], \
				'in_source': [FR_IND, MEM_IND], \
				'deriv_F': [cosine_sim_dkeys, cosine_sim_dmem] })

# F3
F3_IND = len(LAYERS)
F3_shape = (2,L2_shape[0])
L3_shape = (F3_shape[0], L2_shape[1])
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L3_shape, \
				'in_shape': [F3_shape, L2_shape], \
				'in_source': [random_function, COS_IND], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# sq
SQ_IND = len(LAYERS)
LAYERS.append({ 'forward_F': sq_points, \
				'out_shape': L3_shape, \
				'in_shape': [L3_shape], \
				'in_source': [F3_IND], \
				'deriv_F': [sq_points_dinput] })

# sum
LAYERS.append({ 'forward_F': sum_points, \
				'out_shape': (1,), \
				'in_shape': [L3_shape], \
				'in_source': [SQ_IND], \
				'deriv_F': [sum_points_dinput] })				


WEIGHTS = init_weights(LAYERS)
WEIGHTS[FR_IND][1] = random_function(LAYERS[FR_IND]['in_shape'][1])  # inputs
WEIGHTS[FW_IND][1] = random_function(LAYERS[FW_IND]['in_shape'][1])  # inputs_prev
check_weights(WEIGHTS, LAYERS)

####
gradient_layer = FW_IND
gradient_arg = 0
assert isinstance(LAYERS[gradient_layer]['in_source'][gradient_arg], int) != True
ref = WEIGHTS[gradient_layer][gradient_arg]

def f(y):
	WEIGHTS_local = copy.deepcopy(WEIGHTS)
	Wy = WEIGHTS_local[gradient_layer][gradient_arg]
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	WEIGHTS_local[gradient_layer][gradient_arg] = Wy.reshape(weights_shape)
	
	OUTPUT = forward_network(LAYERS, WEIGHTS_local)
	
	return OUTPUT[-1][0]

def g(y):
	WEIGHTS_local = copy.deepcopy(WEIGHTS)
	Wy = WEIGHTS_local[gradient_layer][gradient_arg]
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	WEIGHTS_local[gradient_layer][gradient_arg] = Wy.reshape(weights_shape)
	
	OUTPUT = forward_network(LAYERS, WEIGHTS_local)
	DERIVS, WEIGHT_DERIVS = local_derivs(LAYERS, WEIGHTS_local, OUTPUT)
	reverse_network(deriv_top, len(LAYERS)-1, LAYERS, DERIVS, WEIGHT_DERIVS)
	
	return WEIGHT_DERIVS[gradient_layer][gradient_arg].ravel()[i_ind]

np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e4

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	i_ind = np.random.randint(np.prod(ref.shape))
	y = ref.ravel()[i_ind]
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

