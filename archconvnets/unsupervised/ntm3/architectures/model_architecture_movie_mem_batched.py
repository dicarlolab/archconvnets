from ntm_core import *

def init_model():
	LAYERS = []

	N_CONTROLLERS = 16
	N_MEM_SLOTS = 6
	M_LENGTH = 8

	mem_shape = (BATCH_SZ, N_MEM_SLOTS, M_LENGTH)
	
	U_F1_FILTER_SZ = 5
	U_F2_FILTER_SZ = 5
	U_F3_FILTER_SZ = 3
	
	U_F1 = 16
	U_F2 = 16
	U_F3 = 16
	U_FL = 8
	
	A_F1 = 10
	N_TARGET = 16*16
	N_FC = 32*32*3*2
	HEAD_INPUT = 'FL'

	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,1,32,32), init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, HEAD_INPUT, U_F3, init=init)
		
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
		add_sigmoid_F_bias_layer(LAYERS, 'READ_MEM', N_CONTROLLERS, 'R_F', 'MEM-', init=init)
		
		## above mem
		add_sigmoid_F_bias_layer(LAYERS, 'A_F1', N_FC, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'MEM_STACK', N_FC, sum_all=True, init=init)
		
		### sum mem and conv stacks
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PX', N_FC, source=(BATCH_SZ,1,32,32), init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_F3', N_FC, source='F1_MAX', init=init)
		
		add_add_layer(LAYERS, 'STACK_SUM_PRE', ['STACK_SUM_F3', 'MEM_STACK'], init=init)
		add_add_layer(LAYERS, 'STACK_SUM_PRE2', ['STACK_SUM_PRE', 'STACK_SUM_PX'], init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PRE3', N_FC, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PRE4', N_FC, init=init)
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM', N_TARGET, init=init)
		
		#add_add_layer(LAYERS, 'ERR', ['STACK_SUM', -1], scalar=-1, init=init)
		#add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM', -1], init=init)
		
		add_sum_layer(LAYERS,'ERR_SUM',init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['MEM', 'R_F', 'W_F'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

