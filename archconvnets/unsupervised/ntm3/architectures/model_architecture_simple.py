from ntm_core import *

def init_model():
	LAYERS = []

	N_CONTROLLERS = 2
	N_MEM_SLOTS = 6
	M_LENGTH = 3

	mem_shape = (BATCH_SZ, N_MEM_SLOTS, M_LENGTH)
	
	U_F1_FILTER_SZ = 5
	U_F2_FILTER_SZ = 5
	U_F3_FILTER_SZ = 3
	
	U_F1 = 16
	U_F2 = 16
	U_F3 = 16
	U_FL = 8
	
	A_F1 = 10
	N_TARGET = 32
	HEAD_INPUT = 'FL'

	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,3,32,32), init=init)
		
		add_filter_sum_layer(LAYERS, 'F1S', 3, init=init)

		add_pearson_layer(LAYERS, 'ERR', ['F1S', -1], init=init)
		add_sum_layer(LAYERS,'ERR_SUM',init=init)
		

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = [] #find_layer(LAYERS, ['MEM'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

