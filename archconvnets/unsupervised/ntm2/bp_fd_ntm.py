import numpy as np
import copy
import time
import scipy.optimize

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
############
# cosine similarity between each controller's key and memory vector

def cosine_sim_dmem(args):
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
	keys, mem = args
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	return numer / denom # [n_controllers, n_mem_slots]

def sum_points(args):
	return args[0].sum()[np.newaxis]

def sum_points_dinput(args):
	return np.ones(tuple(np.concatenate(((1,), args[0].shape))))

def sq_points(args):
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
			assert L['deriv_F'][arg](args).shape == expected_shape
		
		# check if other layers claim to produce expected inputs
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_shape'][arg] == LAYERS[L['in_source'][arg]]['out_shape'], '%i %i' % (layer_ind, arg)
				
		# check if layers are ordered (no inputs to this layer come after this one in the list)
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_source'][arg] < layer_ind

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
			DERIVS[layer_ind][arg] = L['deriv_F'][arg](args)
	return DERIVS, WEIGHT_DERIVS

def reverse_network(deriv_above, layer_ind, LAYERS, DERIVS, WEIGHT_DERIVS):
	assert layer_ind >= 0
	L = LAYERS[layer_ind]
	N_ARGS = len(L['in_shape'])
	
	for arg in range(N_ARGS):
		deriv_above_new = mult_partials(deriv_above, layer_ind, arg, LAYERS, DERIVS)
		src = LAYERS[layer_ind]['in_source'][arg]
		if isinstance(src, int) and src != -1: # go back farther
			reverse_network(deriv_above_new, src, LAYERS, DERIVS, WEIGHT_DERIVS)
		else:
			WEIGHT_DERIVS[layer_ind][arg] = deriv_above_new[0]

#############
N_CONTROLLERS = 16
M_LENGTH = 6
N_MEM_SLOTS = 8

deriv_top = np.ones((1,1))

LAYERS = []

# F1
F1_IND = len(LAYERS)
F1_shape = (N_CONTROLLERS,4)
x_shape = (4,M_LENGTH)
L1_shape = (F1_shape[0], x_shape[1])
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L1_shape, \
				'in_shape': [F1_shape, x_shape], \
				'in_source': [random_function, -1], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# "mem"
MEM_IND = len(LAYERS)
Fw_shape = (N_MEM_SLOTS,8)
m_shape = (8,M_LENGTH)
mem_shape = (N_MEM_SLOTS, M_LENGTH)
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': mem_shape, \
				'in_shape': [Fw_shape, m_shape], \
				'in_source': [random_function, -1], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# cosine
COS_IND = len(LAYERS)
mem_shape = (N_MEM_SLOTS, M_LENGTH)
L2_shape = (N_CONTROLLERS, N_MEM_SLOTS)
LAYERS.append({ 'forward_F': cosine_sim, \
				'out_shape': L2_shape, \
				'in_shape': [L1_shape, mem_shape], \
				'in_source': [F1_IND, MEM_IND], \
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
WEIGHTS[0][1] = random_function(LAYERS[0]['in_shape'][1])  # inputs
WEIGHTS[1][1] = random_function(LAYERS[1]['in_shape'][1])  # mem

gradient_layer = 1
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
