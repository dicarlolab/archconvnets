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
		# below
		add_linear_F_bias_layer(LAYERS, 'F1', U_F1, (2, 1), init=init)
		add_linear_F_bias_layer(LAYERS, 'F2', U_F2, init=init)
		add_linear_F_bias_layer(LAYERS, HEAD_INPUT, U_F3, init=init)
		
		## above
		add_relu_F_bias_layer(LAYERS, 'A_F1', A_F1, init=init)
		add_linear_F_bias_layer(LAYERS, 'A_F2', A_F2, init=init)
		add_sum_layer(LAYERS, 'SUM', init=init)
		
		add_add_layer(LAYERS, 'ERR', ['SUM', -1], scalar=-1, init=init)
		
		#####
		
		add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
		add_sum_layer(LAYERS, 'SUM_ERR', init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','F3','', 'A_F1', 'A_F2']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

