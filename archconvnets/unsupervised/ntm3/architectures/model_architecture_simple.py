from ntm_core import *

def init_model():
	LAYERS = []

	N_CONTROLLERS = 16
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
		# keys: N_CONTROLLERS, M_LENGTH
		# mem: N_MEM_SLOTS, M_LENGTH
		# out: N_CONTROLLERS, N_MEM_SLOTS
		
		add_linear_F_layer(LAYERS, 'F1', N_CONTROLLERS, (BATCH_SZ, 3, M_LENGTH), init=init)
		add_linear_F_layer(LAYERS, 'F1b', N_CONTROLLERS, init=init)
		add_linear_F_layer(LAYERS, 'F2', N_CONTROLLERS, (BATCH_SZ, 5, M_LENGTH), init=init)
		#add_focus_keys_layer(LAYERS, 'F3', ['F1','F2'], batch_imgs=True, init=init)
		#add_focus_keys_layer(LAYERS, 'F32', ['F1','F2'], batch_imgs=True, init=init)
		
		add_add_layer(LAYERS, 'F3s', ['F1b', 'F2-'], init=init)
		
		'''add_linear_F_bias_layer(LAYERS, 'F1', 3, (BATCH_SZ, 3, 2), batch_imgs=True, init=init)
		add_linear_F_bias_layer(LAYERS, 'F2', 3, (BATCH_SZ, 5, 7), batch_imgs=True, init=init)
		add_dotT_layer(LAYERS, 'F3', ['F1','F2'], batch_imgs=True, init=init)
		add_dotT_layer(LAYERS, 'F32', ['F1','F2'], batch_imgs=True, init=init)
		add_add_layer(LAYERS, 'F3s', ['F3', 'F32-'], init=init)'''
		
		add_pearson_layer(LAYERS, 'ERR', ['F3s', -1], init=init)
		add_sum_layer(LAYERS,'ERR_SUM',init=init)
		

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['F2'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

