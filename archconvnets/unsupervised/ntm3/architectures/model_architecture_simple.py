from ntm_core import *

def init_model():
	LAYERS = []

	N_CONTROLLERS = 4
	N_MEM_SLOTS = 6
	M_LENGTH = 4

	mem_shape = (BATCH_SZ, N_MEM_SLOTS, M_LENGTH)
	
	U_F1_FILTER_SZ = 5
	U_F2_FILTER_SZ = 5
	U_F3_FILTER_SZ = 3
	
	U_F1 = 48
	U_F2 = 48
	U_F3 = 48
	U_FL = 8
	
	A_F1 = 10
	N_TARGET = 32*32*3
	HEAD_INPUT = 'FL'

	for init in [0,1]:
		# below
		'''add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,3,32,32), init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		add_conv_layer(LAYERS, 'F2', U_F2, U_F2_FILTER_SZ, init=init)
		add_max_pool_layer(LAYERS, 'F2_MAX', init=init)
		add_conv_layer(LAYERS, 'F3', U_F3, U_F3_FILTER_SZ, init=init)'''
		
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,3,32,32), init=init)
		
		#add_linear_F_bias_layer(LAYERS, 'F1', U_F1, (BATCH_SZ, 4, 4,3), init=init)
		
		add_linear_F_bias_layer(LAYERS, HEAD_INPUT, U_F3, init=init)
		
		#add_linear_F_bias_layer(LAYERS, HEAD_INPUT, U_F3, (BATCH_SZ, 4, 4, 3), init=init)
		
		add_linear_F_bias_layer(LAYERS, 'MEM', (N_MEM_SLOTS, M_LENGTH), source=HEAD_INPUT, squeeze=True, init=init)
		
		# content
		add_linear_F_bias_layer(LAYERS, 'R_KEY', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
		add_cosine_sim_layer(LAYERS, 'R_CONTENT', ['R_KEY', 'MEM'], mem_shape, init=init)
		
		# interpolate
		add_sigmoid_F_bias_layer(LAYERS, 'R_IN_GATE', N_CONTROLLERS, HEAD_INPUT, init=init)
		add_interpolate_layer(LAYERS, 'R_IN_PRE', ['R_IN_GATE', 'R_CONTENT', 'R_F-'], init=init)
		add_softmax_layer(LAYERS, 'R_F', init=init)
		
	
	
		add_linear_F_bias_layer(LAYERS, 'Sa', 6, source=HEAD_INPUT, init=init)
		add_pearson_layer(LAYERS, 'ERR', ['Sa', -1], init=init)
		add_sum_layer(LAYERS,'ERR_SUM',init=init)
		

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['R_F'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

