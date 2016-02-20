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
		add_sigmoid_F_bias_layer(LAYERS, 'T1', N_CONTROLLERS, (BATCH_SZ, 4, 3), init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'T2', N_CONTROLLERS, (BATCH_SZ, 4, 5), init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'T3', 3, (BATCH_SZ, 4, 5), init=init)
		
		#add_dotT_layer(LAYERS, 'S1', ['T1','T2'], init=init)
		#add_add_layer(LAYERS, 'S2', ['T3','S1'], init=init)
		
		add_sq_points_layer(LAYERS, 'Sa', init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['Sa', -1], init=init)
		add_sum_layer(LAYERS,'ERR_SUM',init=init)
		

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []#find_layer(LAYERS, ['T2', 'S1'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

