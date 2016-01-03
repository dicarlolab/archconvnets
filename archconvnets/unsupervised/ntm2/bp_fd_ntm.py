import numpy as np
import copy
import time
import scipy.optimize
from ntm_gradients import *
from ntm_core import *

N_FRAMES = 14
M_LENGTH = 6
N_MEM_SLOTS = 8

#############
LAYERS = []

add_linear_F_layer(LAYERS, 'FW', N_MEM_SLOTS, (8, M_LENGTH))
add_add_layer(LAYERS, 'MEM', ['FW', 'MEM'])
add_focus_keys_layer(LAYERS, 'FM', ['MEM', -1])
add_linear_F_layer(LAYERS, 'F3', 25)
add_sum_layer(LAYERS, 'SUM')

################
FW_IND = find_layer(LAYERS, 'FW')
FM_IND = find_layer(LAYERS, 'FM')
MEM_IND = find_layer(LAYERS, 'MEM')

WEIGHTS = init_weights(LAYERS)
xt = 1e2*random_function(np.concatenate(((N_FRAMES,), LAYERS[FW_IND]['in_shape'][1]))) 
WEIGHTS[FW_IND][1] = xt[0]  # inputs
WEIGHTS[FM_IND][1] = random_function(LAYERS[FM_IND]['in_shape'][1])   # inputs
check_weights(WEIGHTS, LAYERS)

OUTPUT_PREV = [None] * len(LAYERS)
OUTPUT_PREV[MEM_IND] = random_function(LAYERS[MEM_IND]['out_shape'])
check_output_prev(OUTPUT_PREV, LAYERS)

################
gradient_layer = find_layer(LAYERS, 'F3')
gradient_arg = 0
assert isinstance(LAYERS[gradient_layer]['in_source'][gradient_arg], int) != True, 'derivative of intermediate layer'
ref = WEIGHTS[gradient_layer][gradient_arg]

def f(y):
	WEIGHTS_local = copy.deepcopy(WEIGHTS); OUTPUT_PREV_local = copy.deepcopy(OUTPUT_PREV)
	Wy = WEIGHTS_local[gradient_layer][gradient_arg]
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	WEIGHTS_local[gradient_layer][gradient_arg] = Wy.reshape(weights_shape)
	
	for frame in range(N_FRAMES):
		WEIGHTS_local[FW_IND][1] = xt[frame]  # inputs
		
		OUTPUT_PREV_local = forward_network(LAYERS, WEIGHTS_local, OUTPUT_PREV_local)
	
	#print OUTPUT_PREV_local[-1][0]
	return OUTPUT_PREV_local[-1][0]

def g(y):
	WEIGHTS_local = copy.deepcopy(WEIGHTS); OUTPUT_PREV_local = copy.deepcopy(OUTPUT_PREV)
	Wy = WEIGHTS_local[gradient_layer][gradient_arg]
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	WEIGHTS_local[gradient_layer][gradient_arg] = Wy.reshape(weights_shape)
	
	PARTIALS_PREV = init_partials(LAYERS)	
	for frame in range(N_FRAMES):
		WEIGHTS_local[FW_IND][1] = xt[frame]  # inputs
		
		OUTPUT = forward_network(LAYERS, WEIGHTS_local, OUTPUT_PREV_local)
		
		LOCAL_DERIVS = local_derivs(LAYERS, WEIGHTS_local, OUTPUT, OUTPUT_PREV_local)
		WEIGHT_DERIVS = reverse_network(deriv_top, len(LAYERS)-1, LAYERS, LOCAL_DERIVS, PARTIALS_PREV)
		
		# update partials_prev
		MEM_WEIGHT_DERIVS = reverse_mem_network(MEM_IND, LAYERS, LOCAL_DERIVS, PARTIALS_PREV)
		PARTIALS_PREV = copy_partials(MEM_IND, LAYERS, PARTIALS_PREV, MEM_WEIGHT_DERIVS)
		OUTPUT_PREV_local = copy.deepcopy(OUTPUT)
	
	return WEIGHT_DERIVS[gradient_layer][gradient_arg].ravel()[i_ind]

np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e7

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

