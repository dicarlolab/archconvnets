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
F_IND = {}; 

LAYERS = []

N_F1 = 12
N_F2 = 7
N_F3 = 9
HEAD_INPUT = 'F3'

for init in [0,1]:
	# below
	F1_IND = add_linear_F_layer(LAYERS, 'F1', N_F1, (2, 1), init=init)
	add_linear_F_layer(LAYERS, 'F2', N_F2, init=init)
	add_linear_F_layer(LAYERS, HEAD_INPUT, N_F3, init=init)
	
	for RW in ['R', 'W']:
		# content
		add_linear_F_layer(LAYERS, RW+'_KEY', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
		add_cosine_sim_layer(LAYERS, RW+'_CONTENT', [RW+'_KEY', 'MEM-'], mem_shape, init=init)
		add_linear_F_layer(LAYERS, RW+'_BETA', N_CONTROLLERS, HEAD_INPUT, init=init)
		add_focus_keys_layer(LAYERS, RW+'_CONTENT_FOCUSED', [RW+'_CONTENT', RW+'_BETA'], init=init)
		add_softmax_layer(LAYERS, RW+'_CONTENT_SM', init=init)
		
		# interpolate
		add_linear_F_layer(LAYERS, RW+'_IN_GATE_PRE', N_CONTROLLERS, HEAD_INPUT, init=init)
		add_sigmoid_layer(LAYERS, RW+'_IN_GATE', init=init)
		add_interpolate_layer(LAYERS, RW+'_IN_PRE', [RW+'_IN_GATE', RW+'_CONTENT_SM', RW+'_F-'], init=init)
		add_softmax_layer(LAYERS, RW+'_IN', init=init)
		
		# shift
		add_linear_F_layer(LAYERS, RW+'_SHIFT_PRE', (N_CONTROLLERS, N_SHIFTS), HEAD_INPUT, init=init)
		add_softmax_layer(LAYERS, RW+'_SHIFT', init=init)
		add_shift_w_layer(LAYERS, RW+'_SHIFTED', [RW+'_SHIFT', RW+'_IN'], init=init)
		
		# sharpen
		add_linear_F_layer(LAYERS, RW+'_GAMMA_PRE', N_CONTROLLERS, HEAD_INPUT, init=init)
		add_relu_layer(LAYERS, RW+'_GAMMA', init=init)
		F_IND[RW] = add_sharpen_layer(LAYERS, RW+'_F', [RW+'_SHIFTED', RW+'_GAMMA'], init=init)
	
	# erase/add output
	add_linear_F_layer(LAYERS, 'ERASE_PRE', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
	add_sigmoid_layer(LAYERS, 'ERASE', init=init)
	
	add_linear_F_layer(LAYERS, 'ADD_PRE', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
	add_sigmoid_layer(LAYERS, 'ADD', init=init)
	
	add_dotT_layer(LAYERS, 'ERASE_HEAD', ['W_F', 'ERASE'], init=init)
	add_dotT_layer(LAYERS, 'ADD_HEAD', ['W_F', 'ADD'], init=init)
	
	# mem = mem_prev * (1 - dotT(W_F, ERASE) + dotT(W_F, ADD)
	
	# mem_prev*erase
	add_mult_layer(LAYERS, 'MEM_ERASE', ['ERASE_HEAD', 'MEM-'], init=init)
	# mem_prev -= mem_prev*erase
	add_add_layer(LAYERS, 'MEM_ERASED', ['MEM-', 'MEM_ERASE'], scalar=-1, init=init)
	# mem_prev += add
	MEM_IND = add_add_layer(LAYERS, 'MEM', ['ADD_HEAD', 'MEM_ERASED'], init=init)
	
	add_sq_points_layer(LAYERS, 'SQ', init=init)
	add_sum_layer(LAYERS, 'SUM', init=init)

MEM_INDS = [MEM_IND, F_IND['R'], F_IND['W']]

check_network(LAYERS)

################ init weights and inputs

WEIGHTS = init_weights(LAYERS)
x1t = random_function(np.concatenate(((N_FRAMES,), LAYERS[F1_IND]['in_shape'][1])))
PREV_VALS = random_function_list(LAYERS, MEM_INDS)

################ which gradient to test
gradient_layer = F1_IND
gradient_arg = 0

def f(y):
	OUTPUT = None
	OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
	
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
	MEM_DERIVS = [None]*len(MEM_INDS)
	OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
	for frame in range(N_FRAMES):
		set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs
		
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		LOCAL_DERIVS = local_derivs(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, LOCAL_DERIVS)
		WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, WEIGHT_DERIVS)
		
		# update partials_prev
		MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
		PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
		
		OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
		
	z = return_buffer(WEIGHT_DERIVS[gradient_layer][gradient_arg]).ravel()[i_ind]
	
	free_list_list(MEM_DERIVS)
	free_partials(PARTIALS_PREV)
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
