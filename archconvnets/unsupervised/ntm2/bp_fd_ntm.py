import numpy as np
import time
import scipy.optimize
from ntm_core import *

N_FRAMES = 5
N_CONTROLLERS = 16
N_MEM_SLOTS = 6
M_LENGTH = 8

mem_shape = (N_MEM_SLOTS, M_LENGTH)

free_all_buffers()

############# init layers
LAYERS = []

N_F1 = 12
N_F2 = 7
N_F3 = 9
HEAD_INPUT = 'F3'

for init in [0,1]:
	F1_IND = add_linear_F_layer(LAYERS, 'F1', N_F1, (2, 10), init=init)
	F2_IND = add_linear_F_layer(LAYERS, 'F2', N_F2, init=init)
	F3_IND = add_linear_F_layer(LAYERS, HEAD_INPUT, N_F3, init=init)

	# read
	##RKEY_IND = add_linear_F_layer(LAYERS, 'RKEY', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)

	RBETA_IND = add_linear_F_layer(LAYERS, 'RBETA', N_CONTROLLERS, HEAD_INPUT, init=init)

	FT_IND = add_linear_F_layer(LAYERS, 'FT', M_LENGTH, (3,10), init=init)

	RCONTENT_IND = add_cosine_sim_layer(LAYERS, 'RCONTENT', ['RBETA', 'FT'], init=init)
	RCONTENT_SUM_IND = add_add_layer(LAYERS, 'RCONTENT_SUM', ['RCONTENT', 'MEM'], init=init)

	MEM_IND = add_add_layer(LAYERS, 'MEM', ['RCONTENT_SUM', 'MEM'], -1, init=init)
	SQ_IND = add_sq_points_layer(LAYERS, 'SQ', init=init)
	add_sum_layer(LAYERS, 'SUM', init=init)

'''F1_IND = add_linear_F_layer(LAYERS, 'F1', N_MEM_SLOTS, (N_MEM_SLOTS, 5))
F2_IND = add_linear_F_layer(LAYERS, 'F2', N_MEM_SLOTS, (N_MEM_SLOTS, 5))
F3_IND = add_mult_layer(LAYERS, 'F3', ['F1','F2'])
MEM_IND = add_add_layer(LAYERS, 'MEM', ['F3', 'MEM'], -1)
SQ_IND = add_sq_points_layer(LAYERS, 'SQ')
add_sum_layer(LAYERS, 'SUM')'''

'''F1_IND = add_linear_F_layer(LAYERS, 'F1', N_MEM_SLOTS, (N_MEM_SLOTS, 3))
F2_IND = add_linear_F_layer(LAYERS, 'F2', N_MEM_SLOTS, (N_MEM_SLOTS, 5))
F3_IND = add_dotT_layer(LAYERS, 'F3', ('F1','F2'))
MEM_IND = add_add_layer(LAYERS, 'MEM', ['F3', 'MEM'], -1)
SQ_IND = add_sq_points_layer(LAYERS, 'SQ')
add_sum_layer(LAYERS, 'SUM')'''

'''F1_IND = add_linear_F_layer(LAYERS, 'F1', N_MEM_SLOTS, (N_MEM_SLOTS, 5))
F1SQ_IND = add_sq_points_layer(LAYERS, 'F1SQ')
F2_IND = add_linear_F_layer(LAYERS, 'F2', N_MEM_SLOTS, (N_MEM_SLOTS, 1))
FS_IND = add_sharpen_layer(LAYERS, 'FS', ['F1SQ', 'F2'])
MEM_IND = add_add_layer(LAYERS, 'MEM', ['FS', 'MEM'])
SQ_IND = add_sq_points_layer(LAYERS, 'SQ')
add_sum_layer(LAYERS, 'SUM')'''

'''F1_IND = add_linear_F_layer(LAYERS, 'F1', N_MEM_SLOTS, (N_MEM_SLOTS, 5))
F2_IND = add_linear_F_layer(LAYERS, 'F2', N_MEM_SLOTS, (N_MEM_SLOTS, 5))
F1S_IND = add_softmax_layer(LAYERS, 'F1S', 'F1')
F3_IND = add_add_layer(LAYERS, 'F3', ['F1S', 'F2'])
MEM_IND = add_add_layer(LAYERS, 'MEM', ['F3', 'MEM'])
SQ_IND = add_sq_points_layer(LAYERS, 'SQ')
add_sum_layer(LAYERS, 'SUM')'''

'''F1_IND = add_linear_F_layer(LAYERS, 'F1', N_MEM_SLOTS, (N_MEM_SLOTS, 1))
F2_IND = add_linear_F_layer(LAYERS, 'F2', N_MEM_SLOTS, (N_MEM_SLOTS, M_LENGTH))
F2S_IND = add_softmax_layer(LAYERS, 'F2S', 'F2')
F3_IND = add_linear_F_layer(LAYERS, 'F3', N_MEM_SLOTS, (N_MEM_SLOTS, M_LENGTH))
FI_IND = add_interpolate_layer(LAYERS, 'FI', ['F1','F2S','F3'])
MEM_IND = add_add_layer(LAYERS, 'MEM', ['FI', 'MEM'])
add_sum_layer(LAYERS, 'SUM')'''

'''FW_IND = add_linear_F_layer(LAYERS, 'FW', N_MEM_SLOTS, (8, M_LENGTH))
#R1_IND = add_relu_layer(LAYERS,'R1')
#S1_IND = add_sigmoid_layer(LAYERS, 'S1')
S2_IND = add_shift_w_layer(LAYERS,'S2', [random_function, 'FW'])
#FS_IND = add_sharpen_layer(LAYERS, 'FS', ['S1', random_function_1])
MEM_IND = add_add_layer(LAYERS, 'MEM', ['S2', 'MEM'])
FM_IND = add_focus_keys_layer(LAYERS, 'FM', ['MEM', random_function])
add_linear_F_layer(LAYERS, 'F3', 25)
add_sum_layer(LAYERS, 'SUM')'''

################ init weights and inputs

WEIGHTS = init_weights(LAYERS)
x1t = random_function(np.concatenate(((N_FRAMES,), LAYERS[F1_IND]['in_shape'][1])))
x2t = random_function(np.concatenate(((N_FRAMES,), LAYERS[FT_IND]['in_shape'][1])))
#x3t = random_function(np.concatenate(((N_FRAMES,), LAYERS[F3_IND]['in_shape'][1])))
mem_init = random_function(LAYERS[MEM_IND]['out_shape'])

DERIV_TOP = init_buffer(np.ones((1,1), dtype='single'))


################ which gradient to test
gradient_layer = F1_IND
gradient_arg = 0

def f(y):
	OUTPUT = None; OUTPUT_PREV = [None] * len(LAYERS)
	OUTPUT_PREV[MEM_IND] = init_buffer(mem_init)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	for frame in range(N_FRAMES):
		set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs
		set_buffer(x2t[frame], WEIGHTS[FT_IND][1])  # inputs
		#set_buffer(x3t[frame], WEIGHTS[F3_IND][1])  # inputs
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	z = return_buffer(OUTPUT[-1])[0]
	free_list(OUTPUT)
	free_list(OUTPUT_PREV)
	return z

def g(y):
	OUTPUT = None; LOCAL_DERIVS = None; WEIGHT_DERIVS = None
	OUTPUT_PREV = [None] * len(LAYERS); MEM_WEIGHT_DERIVS = None
	OUTPUT_PREV[MEM_IND] = init_buffer(mem_init)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	PARTIALS_PREV = init_partials(LAYERS)
	for frame in range(N_FRAMES):
		set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs
		set_buffer(x2t[frame], WEIGHTS[FT_IND][1])  # inputs
		#set_buffer(x3t[frame], WEIGHTS[F3_IND][1])  # inputs
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		LOCAL_DERIVS = local_derivs(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, LOCAL_DERIVS)
		WEIGHT_DERIVS = reverse_network(DERIV_TOP, len(LAYERS)-1, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, WEIGHT_DERIVS)
		
		# update partials_prev
		MEM_WEIGHT_DERIVS = reverse_mem_network(MEM_IND, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, MEM_WEIGHT_DERIVS)
		PARTIALS_PREV = copy_partials(MEM_IND, LAYERS, PARTIALS_PREV, MEM_WEIGHT_DERIVS)
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
eps = np.sqrt(np.finfo(np.float).eps)*1e5

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
