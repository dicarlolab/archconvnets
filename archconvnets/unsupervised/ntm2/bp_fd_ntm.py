import numpy as np
import time
import scipy.optimize
from ntm_core import *

N_FRAMES = 3
N_CONTROLLERS = 16
N_MEM_SLOTS = 6
M_LENGTH = 8
N_SHIFTS = 3

mem_shape = (N_MEM_SLOTS, M_LENGTH)

free_all_buffers()

############# init layers
KEY_IND = {}; CONTENT_IND = {}; 
BETA_IND = {}; CONTENT_FOCUSED_IND = {}; 
CONTENT_SM_IND = {}; IN_GATE_PRE_IND = {}; 
IN_GATE_IND = {}; IN_PRE_IND = {}; 
IN_IND = {}; SHIFT_PRE_IND = {}; 
SHIFT_IND = {}; SHIFTED_IND = {}; 
GAMMA_PRE_IND = {}; GAMMA_IND = {}; 
GAMMA_PRE_IND = {}; F_IND = {}; 

LAYERS = []

N_F1 = 12
N_F2 = 7
N_F3 = 9
HEAD_INPUT = 'F3'

for init in [0,1]:
	# below
	F1_IND = add_linear_F_layer(LAYERS, 'F1', N_F1, (2, 1), init=init)
	F2_IND = add_linear_F_layer(LAYERS, 'F2', N_F2, init=init)
	F3_IND = add_linear_F_layer(LAYERS, HEAD_INPUT, N_F3, init=init)

	R_T_F_IND = add_linear_F_layer(LAYERS, 'R_T_F', (N_CONTROLLERS, N_MEM_SLOTS), HEAD_INPUT, squeeze=True, init=init)
	MEM_T_IND = add_linear_F_layer(LAYERS, 'MEM_T', mem_shape, HEAD_INPUT, squeeze=True, init=init)
	
	for RW in ['R', 'W']:
		# content
		KEY_IND[RW] = add_linear_F_layer(LAYERS, RW+'_KEY', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, squeeze=True, init=init)
		CONTENT_IND[RW] = add_cosine_sim_layer(LAYERS, RW+'_CONTENT', [RW+'_KEY', 'MEM-'], mem_shape, init=init)
		BETA_IND[RW] = add_linear_F_layer(LAYERS, RW+'_BETA', N_CONTROLLERS, HEAD_INPUT, init=init)
		CONTENT_FOCUSED_IND[RW] = add_focus_keys_layer(LAYERS, RW+'_CONTENT_FOCUSED', [RW+'_CONTENT', RW+'_BETA'], init=init)
		CONTENT_SM_IND[RW] = add_softmax_layer(LAYERS, RW+'_CONTENT_SM', init=init)
		
		# interpolate
		IN_GATE_PRE_IND[RW] = add_linear_F_layer(LAYERS, RW+'_IN_GATE_PRE', N_CONTROLLERS, HEAD_INPUT, init=init)
		IN_GATE_IND[RW] = add_sigmoid_layer(LAYERS, RW+'_IN_GATE', init=init)
		IN_PRE_IND[RW] = add_interpolate_layer(LAYERS, RW+'_IN_PRE', [RW+'_IN_GATE', RW+'_CONTENT_SM', 'R_T_F'], init=init)
		IN_IND[RW] = add_softmax_layer(LAYERS, RW+'_IN', init=init)
		
		# shift
		SHIFT_PRE_IND[RW] = add_linear_F_layer(LAYERS, RW+'_SHIFT_PRE', (N_CONTROLLERS, N_SHIFTS), HEAD_INPUT, init=init)
		SHIFT_IND[RW] = add_softmax_layer(LAYERS, RW+'_SHIFT', init=init)
		SHIFTED_IND[RW] = add_shift_w_layer(LAYERS, RW+'_SHIFTED', [RW+'_SHIFT', RW+'_IN'], init=init)
		
		# sharpen
		GAMMA_PRE_IND[RW] = add_linear_F_layer(LAYERS, RW+'_GAMMA_PRE', N_CONTROLLERS, HEAD_INPUT, init=init)
		GAMMA_IND[RW] = add_relu_layer(LAYERS, RW+'_GAMMA', init=init)
		F_IND[RW] = add_sharpen_layer(LAYERS, RW+'_F', [RW+'_SHIFTED', RW+'_GAMMA'], init=init)
	
	# erase/add output
	ERASE_PRE_IND = add_linear_F_layer(LAYERS, 'ERASE_PRE', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
	ERASE_IND = add_sigmoid_layer(LAYERS, 'ERASE', init=init)
	
	ADD_PRE_IND = add_linear_F_layer(LAYERS, 'ADD_PRE', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
	ADD_IND = add_sigmoid_layer(LAYERS, 'ADD', init=init)
	
	ERASE_HEAD_IND = add_dotT_layer(LAYERS, 'ERASE_HEAD', ['W_F', 'ERASE'], init=init)
	ADD_HEAD_IND = add_dotT_layer(LAYERS, 'ADD_HEAD', ['W_F', 'ADD'], init=init)
	
	
	# mem = mem_prev * (1 - dotT(W_F, ERASE) + dotT(W_F, ADD)
	
	# mem_prev*erase
	MEM_ERASE_IND = add_mult_layer(LAYERS, 'MEM_ERASE', ['ERASE_HEAD', 'MEM-'], init=init)
	# mem_prev -= mem_prev*erase
	MEM_ERASED_IND = add_add_layer(LAYERS, 'MEM_ERASED', ['MEM-', 'MEM_ERASE'], scalar=-1, init=init)
	# mem_prev += add
	MEM_IND = add_add_layer(LAYERS, 'MEM', ['ADD_HEAD', 'MEM_ERASED'], init=init)
	
	
	SQ_IND = add_sq_points_layer(LAYERS, 'SQ', init=init)
	add_sum_layer(LAYERS, 'SUM', init=init)


check_network(LAYERS)

################ init weights and inputs

WEIGHTS = init_weights(LAYERS)
x1t = random_function(np.concatenate(((N_FRAMES,), LAYERS[F1_IND]['in_shape'][1])))
mem_init = random_function(LAYERS[MEM_IND]['out_shape'])

DERIV_TOP = init_buffer(np.ones((1,1), dtype='single'))

################ which gradient to test
gradient_layer = F1_IND
gradient_arg = 0

def f(y):
	OUTPUT = None; OUTPUT_PREV = [None] * len(LAYERS)
	OUTPUT_PREV[MEM_IND] = init_buffer(mem_init)
	#OUTPUT_PREV[F_IND['R']] = init_buffer(R_F_init)
	#OUTPUT_PREV[MEM2_IND] = init_buffer(mem2_init)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	for frame in range(N_FRAMES):
		set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs

		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	z = return_buffer(OUTPUT[-1])[0]
	free_list(OUTPUT)
	free_list(OUTPUT_PREV)
	return z

def g(y):
	OUTPUT = None; LOCAL_DERIVS = None; WEIGHT_DERIVS = None
	OUTPUT_PREV = [None] * len(LAYERS); MEM_WEIGHT_DERIVS = None
	MEM2_WEIGHT_DERIVS = None; R_F_DERIVS = None
	OUTPUT_PREV[MEM_IND] = init_buffer(mem_init)
	#OUTPUT_PREV[F_IND['R']] = init_buffer(R_F_init)
	#OUTPUT_PREV[MEM2_IND] = init_buffer(mem2_init)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	PARTIALS_PREV = init_partials(LAYERS, MEM_IND)
	for frame in range(N_FRAMES):
		set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs
		
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		LOCAL_DERIVS = local_derivs(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, LOCAL_DERIVS)
		WEIGHT_DERIVS = reverse_network(DERIV_TOP, len(LAYERS)-1, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, WEIGHT_DERIVS)
		
		# update partials_prev
		MEM_WEIGHT_DERIVS = reverse_network(None, MEM_IND, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, MEM_WEIGHT_DERIVS, keep_dims=True)
		##R_F_DERIVS = reverse_network(None, R_F_IND, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, R_F_DERIVS, keep_dims=True)
		#MEM2_WEIGHT_DERIVS = reverse_network(None, MEM2_IND, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, MEM2_WEIGHT_DERIVS, keep_dims=True)
		PARTIALS_PREV = copy_partials(MEM_IND, LAYERS, PARTIALS_PREV, MEM_WEIGHT_DERIVS)
		##PARTIALS_PREV = copy_partials(R_F_IND, LAYERS, PARTIALS_PREV, R_F_DERIVS)
		#PARTIALS_PREV = copy_partials(MEM2_IND, LAYERS, PARTIALS_PREV, MEM2_WEIGHT_DERIVS)
		OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
		
	z = return_buffer(WEIGHT_DERIVS[gradient_layer][gradient_arg]).ravel()[i_ind]
	
	free_partials(PARTIALS_PREV)
	free_list(MEM_WEIGHT_DERIVS)
	free_list(LOCAL_DERIVS)
	free_list(OUTPUT)
	free_list(WEIGHT_DERIVS)
	free_list(OUTPUT_PREV)
	return z

assert isinstance(LAYERS[gradient_layer]['in_source'][gradient_arg], int) != True, 'derivative of intermediate layer'
ref = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e6

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
t_start = time.time()
for sample in range(N_SAMPLES):
	i_ind = np.random.randint(np.prod(ref.shape))
	y = ref.ravel()[i_ind]
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std(), time.time() - t_start, GPU
