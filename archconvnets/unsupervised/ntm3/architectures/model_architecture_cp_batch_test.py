from ntm_core import *

BATCH_SZ = 25

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
		# below
		add_linear_F_bias_layer(LAYERS, 'F1', U_F1, (BATCH_SZ, 2, 1), batch_imgs=B, init=init)
		add_linear_F_bias_layer(LAYERS, 'F2', U_F2, batch_imgs=B, init=init)
		add_linear_F_bias_layer(LAYERS, HEAD_INPUT, U_F3, batch_imgs=B, init=init)
		add_linear_F_bias_layer(LAYERS, 'MEM', N_MEM_SLOTS, (BATCH_SZ, 3, M_LENGTH), batch_imgs=B, init=init)
		add_linear_F_bias_layer(LAYERS, 'R_KEY', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, batch_imgs=B, init=init)
		add_cosine_sim_layer(LAYERS, 'R_CONTENT', ['R_KEY', 'MEM'], mem_shape, batch_imgs=B, init=init) ###########
		
		add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
		add_sum_layer(LAYERS, 'SUM_ERR', init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = [] #find_layer(LAYERS, ['MEM', 'R_F', 'W_F'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

