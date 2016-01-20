from ntm_core import *

def init_model():
	LAYERS = []

	N_CONTROLLERS = 16
	N_MEM_SLOTS = 6
	M_LENGTH = 8

	mem_shape = (N_MEM_SLOTS, M_LENGTH)
	
	U_F1 = 48
	U_F2 = 48
	U_F3 = 48
	U_FL = 48
	
	A_F1 = 48
	N_TARGET = 4
	HEAD_INPUT = 'FL'

	for init in [0,1]:
		# below
		add_sigmoid_F_bias_layer(LAYERS, 'F1', U_F1, (N_TARGET, 1), init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'F2', U_F2, init=init)
		add_sigmoid_F_bias_layer(LAYERS, HEAD_INPUT, U_F3, init=init)
		
		## above
		add_sigmoid_F_bias_layer(LAYERS, 'A_F1', A_F1, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'A_F2', N_TARGET, init=init)
		
		#add_pearson_layer(LAYERS, 'ERR', ['A_F2', -1], init=init)
		
		#####
		add_add_layer(LAYERS, 'ERR', ['A_F2', -1], scalar=-1, init=init)
		add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
		add_sum_layer(LAYERS, 'SUM_ERR', init=init)
		

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','FL','', 'A_F1', 'A_F2', 'ERR']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

