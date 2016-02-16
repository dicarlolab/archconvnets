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
		# shift_out: [n_controllers, n_shifts], w_interp: [n_controllers, mem_length]
		
		add_linear_F_bias_layer(LAYERS, 'F1', 3, (8, 4, 2), batch_imgs=True, init=init)
		add_linear_F_bias_layer(LAYERS, 'F2', 3, (8, 5, 7), batch_imgs=True, init=init)
		
		add_dotT_layer(LAYERS, 'F3', ['F1','F2'], batch_imgs=True, init=init)
		
		#add_linear_F_bias_layer(LAYERS, 'F1', 3, (4, 2), init=init)
		#add_linear_F_bias_layer(LAYERS, 'F2', 3, (5, 7), init=init)
		
		#add_dotT_layer(LAYERS, 'F3', ['F1','F2'], init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['F3', -1], batch_imgs=True, init=init)
		add_sum_layer(LAYERS,'ERR_SUM',init=init)
		

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []#find_layer(LAYERS, ['MEM'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

