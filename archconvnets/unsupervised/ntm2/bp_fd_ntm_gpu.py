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
				
# F under
FU_IND2 = len(LAYERS)
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

# FW read
FW_IND = len(LAYERS)
F1_shape = (N_CONTROLLERS,8)
L1_shape = (F1_shape[0], x_shape[1])
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L1_shape, \
				'in_shape': [F1_shape, L0_shape], \
				'in_source': [random_function, FU_IND2], \
				'deriv_F': [linear_F_dF, linear_F_dx] })
				
# mem = mem_prev + Fw
MEM_IND = len(LAYERS)
LAYERS.append({ 'forward_F': add_points, \
				'out_shape': L1_shape, \
				'in_shape': [L1_shape, L1_shape], \
				'in_source': [FW_IND, FR_IND], \
				'deriv_F': [add_points_dinput, add_points_dinput] })

# sum
LAYERS.append({ 'forward_F': sum_points, \
				'out_shape': (1,), \
				'in_shape': [L1_shape], \
				'in_source': [MEM_IND], \
				'deriv_F': [sum_points_dinput] })				

'''# sum
LAYERS.append({ 'forward_F': sum_points, \
				'out_shape': (1,), \
				'in_shape': [L1_shape], \
				'in_source': [FW_IND], \
				'deriv_F': [sum_points_dinput] })'''
				
################
WEIGHTS = init_weights(LAYERS)
xt = random_function(np.concatenate(((N_FRAMES,), LAYERS[FU_IND]['in_shape'][1])))
set_buffer(xt[0], WEIGHTS[FU_IND][1])
set_buffer(xt[0], WEIGHTS[FU_IND2][1])
check_weights(WEIGHTS, LAYERS)

DERIV_TOP = init_buffer(np.ones((1,1), dtype='single'))

PARTIALS_PREV = init_partials(LAYERS)

def free_list(LIST):
	for layer_ind in range(len(LIST)):

		# list of list, ex. partials
		if isinstance(LIST[0][0], list):
			for arg in range(len(LIST[layer_ind])):
				free_buffer(LIST[layer_ind][arg])
		# outputs
		else:
			free_buffer(LIST[layer_ind])

################
gradient_layer = FU_IND2
gradient_arg = 0
assert isinstance(LAYERS[gradient_layer]['in_source'][gradient_arg], int) != True, 'derivative of intermediate layer'
ref = return_buffer(WEIGHTS[gradient_layer][gradient_arg])

def f(y):
	OUTPUT = None; OUTPUT_PREV = [None] * len(LAYERS)
	#OUTPUT_PREV[MEM_IND] = init_buffer(random_function(LAYERS[MEM_IND]['out_shape']))
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	for frame in range(N_FRAMES):
		set_buffer(xt[frame], WEIGHTS[FU_IND][1])  # inputs
		set_buffer(xt[frame], WEIGHTS[FU_IND2][1])  # inputs
		check_weights(WEIGHTS, LAYERS)
		
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV) ###
	
	z = return_buffer(OUTPUT[-1])[0]
	free_list(OUTPUT)
	return z

def g(y):
	OUTPUT = None; LOCAL_DERIVS = None; WEIGHT_DERIVS = None
	OUTPUT_PREV = [None] * len(LAYERS)
	#OUTPUT_PREV[MEM_IND] = init_buffer(random_function(LAYERS[MEM_IND]['out_shape']))
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	for frame in range(N_FRAMES):
		if frame != 0: # todo: reverse_network should zero out weight_derivs before beginning
			free_list(WEIGHT_DERIVS); WEIGHT_DERIVS = None
		
		set_buffer(xt[frame], WEIGHTS[FU_IND][1])  # inputs
		set_buffer(xt[frame], WEIGHTS[FU_IND2][1])  # inputs
		check_weights(WEIGHTS, LAYERS)
		
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		
		LOCAL_DERIVS = local_derivs(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, LOCAL_DERIVS)
		WEIGHT_DERIVS = reverse_network(DERIV_TOP, len(LAYERS)-1, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, WEIGHT_DERIVS)
		
		# update partials_prev
		#MEM_WEIGHT_DERIVS = reverse_mem_network(MEM_IND, LAYERS, LOCAL_DERIVS, PARTIALS_PREV)
		
	z = return_buffer(WEIGHT_DERIVS[gradient_layer][gradient_arg]).ravel()[i_ind]
	
	free_list(LOCAL_DERIVS)
	free_list(OUTPUT)
	free_list(WEIGHT_DERIVS)
	return z

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
