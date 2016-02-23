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
	
	U_F1 = 16
	U_F2 = 16
	U_F3 = 16
	U_FL = 8
	
	N_TARGET = 16*16
	N_FC = 32*32*3*2
	HEAD_INPUT = 'FL'

	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,1,32,32), init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		
		
		### sum mem and conv stacks
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PX', N_FC, source=(BATCH_SZ,1,32,32), init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_F3', N_FC, source='F1_MAX', init=init)
		
		add_add_layer(LAYERS, 'STACK_SUM_PRE', ['STACK_SUM_F3', 'STACK_SUM_PX'], init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PRE2', N_FC, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PRE3', N_FC, init=init)
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM', N_TARGET, init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM', -1], init=init)
		
		add_sum_layer(LAYERS,'ERR_SUM',init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

