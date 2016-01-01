import numpy as np
import copy
import time
import scipy.optimize
from ntm_core_gpu import *
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *

N_FRAMES = 5
N_CONTROLLERS = 16
M_LENGTH = 6
N_MEM_SLOTS = 8
mem_shape = (N_MEM_SLOTS, M_LENGTH)

free_all_buffers()

#############
LAYERS = []

# F under
FU_IND = len(LAYERS)
x_shape = (12,M_LENGTH)
F0_shape = (8, 12)
L0_shape = (8,M_LENGTH)
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L0_shape, \
				'in_shape': [F0_shape, x_shape], \
				'in_source': [random_function, -1], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# FR read
FR_IND = len(LAYERS)
F1_shape = (N_CONTROLLERS,8)
L1_shape = (F1_shape[0], x_shape[1])
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L1_shape, \
				'in_shape': [F1_shape, L0_shape], \
				'in_source': [random_function, FU_IND], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# sum
LAYERS.append({ 'forward_F': sum_points, \
				'out_shape': (1,), \
				'in_shape': [L1_shape], \
				'in_source': [FR_IND], \
				'deriv_F': [sum_points_dinput] })				

################
WEIGHTS = init_weights(LAYERS)
xt = random_function(np.concatenate(((N_FRAMES,), LAYERS[FU_IND]['in_shape'][1])))
set_buffer(xt[0], WEIGHTS[FU_IND][1])
check_weights(WEIGHTS, LAYERS)

OUTPUT_PREV = [None] * len(LAYERS)
#OUTPUT_PREV[MEM_IND] = init_buffer(random_function(LAYERS[MEM_IND]['out_shape']))
check_output_prev(OUTPUT_PREV, LAYERS)

OUTPUT = None; LOCAL_DERIVS = None; WEIGHT_DERIVS = None
DERIV_TOP = init_buffer(np.ones((1,1), dtype='single'))

PARTIALS_PREV = init_partials(LAYERS)
'''OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)

LOCAL_DERIVS = local_derivs(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, LOCAL_DERIVS)
WEIGHT_DERIVS = reverse_network(DERIV_TOP, len(LAYERS)-1, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, WEIGHT_DERIVS)'''

################
gradient_layer = FU_IND
gradient_arg = 0
assert isinstance(LAYERS[gradient_layer]['in_source'][gradient_arg], int) != True, 'derivative of intermediate layer'
ref = return_buffer(WEIGHTS[gradient_layer][gradient_arg])

def f(y):
	OUTPUT_local = copy.deepcopy(OUTPUT); OUTPUT_PREV_local = copy.deepcopy(OUTPUT_PREV)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	for frame in range(N_FRAMES):
		set_buffer(xt[frame], WEIGHTS[FU_IND][1])  # inputs
		
		OUTPUT_PREV_local = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV_local)
	
	return return_buffer(OUTPUT_PREV_local[-1])[0]

def g(y):
	OUTPUT_local = copy.deepcopy(OUTPUT); OUTPUT_PREV_local = copy.deepcopy(OUTPUT_PREV)
	LOCAL_DERIVS_local = copy.deepcopy(LOCAL_DERIVS); WEIGHT_DERIVS_local = copy.deepcopy(WEIGHT_DERIVS)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	for frame in range(N_FRAMES):
		set_buffer(xt[frame], WEIGHTS[FU_IND][1])  # inputs
		
		OUTPUT_local = forward_network(LAYERS, WEIGHTS, OUTPUT_local, OUTPUT_PREV_local)
		
		LOCAL_DERIVS_local = local_derivs(LAYERS, WEIGHTS, OUTPUT_local, OUTPUT_PREV_local, LOCAL_DERIVS)
		WEIGHT_DERIVS_local = reverse_network(DERIV_TOP, len(LAYERS)-1, LAYERS, LOCAL_DERIVS_local, PARTIALS_PREV, WEIGHT_DERIVS_local)
		
	return return_buffer(WEIGHT_DERIVS_local[gradient_layer][gradient_arg]).ravel()[i_ind]

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
