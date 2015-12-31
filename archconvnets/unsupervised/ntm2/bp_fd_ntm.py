import numpy as np
import copy
import time
import scipy.optimize
from ntm_gradients import *
from ntm_core import *

N_FRAMES = 5
N_CONTROLLERS = 16
M_LENGTH = 6
N_MEM_SLOTS = 8
mem_shape = (N_MEM_SLOTS, M_LENGTH)

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

# Fw write
FW_IND = len(LAYERS)
Fw_shape = (N_MEM_SLOTS,8)
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': mem_shape, \
				'in_shape': [Fw_shape, L0_shape], \
				'in_source': [random_function, FU_IND], \
				'deriv_F': [linear_F_dF, linear_F_dx] })
		
# mem = mem_prev + Fw
MEM_IND = len(LAYERS)
LAYERS.append({ 'forward_F': add_points, \
				'out_shape': mem_shape, \
				'in_shape': [mem_shape, mem_shape], \
				'in_source': [FW_IND, MEM_IND], \
				'deriv_F': [add_points_dinput, add_points_dinput] })

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

################
WEIGHTS = init_weights(LAYERS)
xt = random_function(np.concatenate(((N_FRAMES,), LAYERS[FU_IND]['in_shape'][1])))
WEIGHTS[FU_IND][1] = xt[0]  # inputs
check_weights(WEIGHTS, LAYERS)

OUTPUT_PREV = [None] * len(LAYERS)
OUTPUT_PREV[MEM_IND] = random_function(LAYERS[MEM_IND]['out_shape'])
check_output_prev(OUTPUT_PREV, LAYERS)

################
gradient_layer = FU_IND
gradient_arg = 0
assert isinstance(LAYERS[gradient_layer]['in_source'][gradient_arg], int) != True, 'derivative of intermediate layer'
ref = WEIGHTS[gradient_layer][gradient_arg]

def f(y):
	WEIGHTS_local = copy.deepcopy(WEIGHTS); OUTPUT_PREV_local = copy.deepcopy(OUTPUT_PREV)
	Wy = WEIGHTS_local[gradient_layer][gradient_arg]
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	WEIGHTS_local[gradient_layer][gradient_arg] = Wy.reshape(weights_shape)
	
	for frame in range(N_FRAMES):
		WEIGHTS_local[FU_IND][1] = xt[frame]  # inputs
		
		OUTPUT_PREV_local = forward_network(LAYERS, WEIGHTS_local, OUTPUT_PREV_local)
	
	return OUTPUT_PREV_local[-1][0]

def g(y):
	WEIGHTS_local = copy.deepcopy(WEIGHTS); OUTPUT_PREV_local = copy.deepcopy(OUTPUT_PREV)
	Wy = WEIGHTS_local[gradient_layer][gradient_arg]
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	WEIGHTS_local[gradient_layer][gradient_arg] = Wy.reshape(weights_shape)
	
	PARTIALS_PREV = init_partials(LAYERS)	
	for frame in range(N_FRAMES):
		WEIGHTS_local[FU_IND][1] = xt[frame]  # inputs
		
		OUTPUT = forward_network(LAYERS, WEIGHTS_local, OUTPUT_PREV_local)
		
		LOCAL_DERIVS = local_derivs(LAYERS, WEIGHTS_local, OUTPUT, OUTPUT_PREV_local)
		WEIGHT_DERIVS = reverse_network(deriv_top, len(LAYERS)-1, LAYERS, LOCAL_DERIVS, PARTIALS_PREV)
		
		# update partials_prev
		MEM_WEIGHT_DERIVS = reverse_mem_network(MEM_IND, LAYERS, LOCAL_DERIVS, PARTIALS_PREV)
		PARTIALS_PREV = copy_partials(MEM_IND, LAYERS, PARTIALS_PREV, MEM_WEIGHT_DERIVS)
		OUTPUT_PREV_local = copy.deepcopy(OUTPUT)
	
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

