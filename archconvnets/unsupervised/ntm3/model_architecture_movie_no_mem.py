from ntm_core import *

def init_model():
	LAYERS = []

	N_CONTROLLERS = 4
	N_MEM_SLOTS = 6
	M_LENGTH = 4

	mem_shape = (N_MEM_SLOTS, M_LENGTH)
	
	U_F1_FILTER_SZ = 5
	U_F2_FILTER_SZ = 5
	U_F3_FILTER_SZ = 3
	
	U_F1 = 48
	U_F2 = 48
	U_F3 = 48
	U_FL = 8
	
	A_F1 = 10
	N_TARGET = 32*32
	HEAD_INPUT = 'FL'

	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(1,3,32,32), init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		add_conv_layer(LAYERS, 'F2', U_F2, U_F2_FILTER_SZ, init=init)
		add_max_pool_layer(LAYERS, 'F2_MAX', init=init)
		add_conv_layer(LAYERS, 'F3', U_F3, U_F3_FILTER_SZ, init=init)
		
		### sum mem and conv stacks
		add_relu_F_bias_layer(LAYERS, 'STACK_SUM', N_TARGET, source='F3', init=init)
		
		add_add_layer(LAYERS, 'ERR', ['STACK_SUM', -1], scalar=-1, init=init)
		
		#####
		
		add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
		add_sum_layer(LAYERS, 'SUM_ERR', init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','F3','', 'STACK_SUM']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

