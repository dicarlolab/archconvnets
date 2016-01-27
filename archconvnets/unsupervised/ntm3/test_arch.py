from ntm_core import *

N_FRAMES_PRED = 1
N_IN = 64*64*3

LAYERS = []

N_CONTROLLERS = 16
N_MEM_SLOTS = 8
M_LENGTH = 4

mem_shape = (N_MEM_SLOTS, M_LENGTH)

U_F1_FILTER_SZ = 5
U_F2_FILTER_SZ = 5
U_F3_FILTER_SZ = 3

U_F1 = 48
U_F2 = 48
U_F3 = 48
U_FL = 8

A_F0 = 48
A_F1 = 48

N_TARGET = N_IN*N_FRAMES_PRED
HEAD_INPUT = 'FL'

for init in [0,1]:
	# below
	add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(1,3,64,64), init=init)
	add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
	add_conv_layer(LAYERS, 'F2', U_F2, U_F2_FILTER_SZ, init=init)
	add_max_pool_layer(LAYERS, 'F2_MAX', init=init)
	add_conv_layer(LAYERS, 'F3', U_F3, U_F3_FILTER_SZ, init=init)
	add_max_pool_layer(LAYERS, 'F3_MAX', init=init)
	add_linear_F_bias_layer(LAYERS, HEAD_INPUT, U_F3, init=init)
	
	for RW in ['R', 'W']:
		# content
		add_linear_F_bias_layer(LAYERS, RW+'_KEY', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
		add_cosine_sim_layer(LAYERS, RW+'_CONTENT', [RW+'_KEY', 'MEM-'], mem_shape, init=init)
		add_linear_F_bias_layer(LAYERS, RW+'_BETA', N_CONTROLLERS, HEAD_INPUT, init=init)
		add_focus_keys_layer(LAYERS, RW+'_CONTENT_FOCUSED', [RW+'_CONTENT', RW+'_BETA'], init=init)
		add_softmax_layer(LAYERS, RW+'_CONTENT_SM', init=init)
		
		# interpolate
		add_sigmoid_F_bias_layer(LAYERS, RW+'_IN_GATE', N_CONTROLLERS, HEAD_INPUT, init=init)
		add_interpolate_layer(LAYERS, RW+'_IN_PRE', [RW+'_IN_GATE', RW+'_CONTENT_SM', RW+'_F-'], init=init)
		add_softmax_layer(LAYERS, RW+'_IN', init=init)
		
		# shift
		add_linear_F_bias_layer(LAYERS, RW+'_SHIFT_PRE', (N_CONTROLLERS, N_SHIFTS), HEAD_INPUT, init=init)
		add_softmax_layer(LAYERS, RW+'_SHIFT', init=init)
		add_shift_w_layer(LAYERS, RW+'_SHIFTED', [RW+'_SHIFT', RW+'_IN'], init=init)
		
		# sharpen
		add_relu_F_bias_layer(LAYERS, RW+'_GAMMA', N_CONTROLLERS, HEAD_INPUT, init=init)
		add_sharpen_layer(LAYERS, RW+'_F', [RW+'_SHIFTED', RW+'_GAMMA'], init=init)
	
	##### write
	# erase/add output
	add_sigmoid_F_bias_layer(LAYERS, 'ERASE', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'ADD', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
	
	add_dotT_layer(LAYERS, 'ERASE_HEAD', ['W_F', 'ERASE'], init=init)
	add_dotT_layer(LAYERS, 'ADD_HEAD', ['W_F', 'ADD'], init=init)
	
	# mem = mem_prev * (1 - dotT(W_F, ERASE) + dotT(W_F, ADD)
	add_mult_layer(LAYERS, 'MEM_ERASE', ['ERASE_HEAD', 'MEM-'], init=init)
	add_add_layer(LAYERS, 'MEM_ERASED', ['MEM-', 'MEM_ERASE'], scalar=-1, init=init)
	add_add_layer(LAYERS, 'MEM', ['ADD_HEAD', 'MEM_ERASED'], init=init)
	
	##### read
	add_linear_F_bias_layer(LAYERS, 'READ_MEM', N_CONTROLLERS, 'R_F', 'MEM-', init=init)
	
	
	## above mem
	add_sigmoid_F_bias_layer(LAYERS, 'A_F0M', A_F0, init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'A_F1M', A_F1, init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'A_F2M', N_TARGET, init=init)
	
	## above inputs (Max3)
	add_sigmoid_F_bias_layer(LAYERS, 'M3_F0', A_F0, source=HEAD_INPUT, init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'M3_F1', A_F1, init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'M3_F2', N_TARGET, init=init)
	
	## above inputs (conv1)
	add_sigmoid_F_bias_layer(LAYERS, 'C1_F0', A_F0, source='F1', init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'C1_F1', A_F1, init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'C1_F2', N_TARGET, init=init)
	
	## above inputs (max2)
	add_sigmoid_F_bias_layer(LAYERS, 'M2_F0', A_F0, source='F2_MAX', init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'M2_F1', A_F1, init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'M2_F2', N_TARGET, init=init)
	
	add_add_layer(LAYERS, 'STACK_SUM0', ['M3_F2', 'A_F2M'], init=init)
	add_add_layer(LAYERS, 'STACK_SUM1', ['STACK_SUM0', 'M3_F2'], init=init)
	add_add_layer(LAYERS, 'STACK_SUM2', ['STACK_SUM1', 'M2_F2'], init=init)
	
	add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM3', N_TARGET, init=init)
	add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM4', N_TARGET, init=init)
	
	add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM4', -1], init=init)

check_network(LAYERS)

################ init weights and inputs
WEIGHTS = init_weights(LAYERS)
MEM_INDS = find_layer(LAYERS, ['MEM', 'R_F', 'W_F'])
PREV_VALS = random_function_list(LAYERS, MEM_INDS)


