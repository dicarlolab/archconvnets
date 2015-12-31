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
		assert L['forward_F'](args).shape == L['out_shape'], "%i" % (layer)
		
		# check if deriv functions correctly produce correct shapes
		for arg in range(N_ARGS):
			expected_shape = tuple(np.concatenate((L['out_shape'], L['in_shape'][arg])))
			assert L['deriv_F'][arg](args).shape == expected_shape
		
		# check if other layers claim to produce expected inputs
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_shape'][arg] == LAYERS[L['in_source'][arg]]['out_shape']
				
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
			reverse_network(deriv_above_new, layer_ind-1, LAYERS, DERIVS, WEIGHT_DERIVS)
		else:
			WEIGHT_DERIVS[layer_ind][arg] = deriv_above_new[0]

#############
deriv_top = np.ones((1,1))

F1_shape = (3,4)
x_shape = (4,5)
L1_shape = (F1_shape[0], x_shape[1])

F2_shape = (8,L1_shape[0])
L2_shape = (F2_shape[0], L1_shape[1])

F3_shape = (2,L2_shape[0])
L3_shape = (F3_shape[0], L2_shape[1])

LAYERS = []
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L1_shape, \
				'in_shape': [F1_shape, x_shape], \
				'in_source': [random_function, -1], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L2_shape, \
				'in_shape': [F2_shape, L1_shape], \
				'in_source': [random_function, 0], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L3_shape, \
				'in_shape': [F3_shape, L2_shape], \
				'in_source': [random_function, 1], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

LAYERS.append({ 'forward_F': sq_points, \
				'out_shape': L3_shape, \
				'in_shape': [L3_shape], \
				'in_source': [2], \
				'deriv_F': [sq_points_dinput] })

LAYERS.append({ 'forward_F': sum_points, \
				'out_shape': (1,), \
				'in_shape': [L3_shape], \
				'in_source': [3], \
				'deriv_F': [sum_points_dinput] })				


WEIGHTS = init_weights(LAYERS)
WEIGHTS[0][1] = random_function(LAYERS[0]['in_shape'][1])  # inputs

gradient_layer = 1
gradient_arg = 0
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

