from ntm_core import *

def init_model():
	LAYERS = []

	BATCH_SZ = 5
	N_CONTROLLERS = 12
	N_MEM_SLOTS = 6
	M_LENGTH = 8

	mem_shape = (N_MEM_SLOTS, M_LENGTH)

	U_F1 = 12
	U_F2 = 7
	U_F3 = 9
	
	A_F1 = 4
	A_F2 = 7
	HEAD_INPUT = 'F3'

	for init in [0,1]:
		add_conv_layer(LAYERS, 'F3', U_F3, 5, source = -1, imgs_shape=(BATCH_SZ,3,32,32), init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		add_conv_layer(LAYERS, 'F4', U_F3, 5, init=init)
		add_linear_F_bias_layer(LAYERS, 'Sa', U_F3, init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['Sa-', -1], init=init)
		add_sum_layer(LAYERS,'ERR_SUM',init=init)
		

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['Sa'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

