from ntm_core import *

def init_model():
	LAYERS = []
	
	N_C = 32
	
	N_CONTROLLERS = 16
	N_MEM_SLOTS = 6
	M_LENGTH = 8

	mem_shape = (BATCH_SZ, N_MEM_SLOTS, M_LENGTH)

	U_F1 = 12
	U_F2 = 7
	U_F3 = 9
	
	A_F1 = 4
	A_F2 = 7
	HEAD_INPUT = 'F3'

	for init in [0,1]:
		# below
		add_sigmoid_F_bias_layer(LAYERS, 'F1', U_F1, (BATCH_SZ, 2, 1), init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'F2', U_F2, init=init)
		add_sigmoid_F_bias_layer(LAYERS, HEAD_INPUT, U_F3, init=init)
		
		add_linear_F_bias_layer(LAYERS, 'IN', N_C, HEAD_INPUT, init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'IN_GATE', N_C, HEAD_INPUT, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'FORGET_GATE', N_C, HEAD_INPUT, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'OUT_GATE', N_C, HEAD_INPUT, init=init)
		
		add_mult_layer(LAYERS, 'IN_MULT', ['IN', 'IN_GATE'], init=init)
		add_mult_layer(LAYERS, 'CEC_MULT', ['FORGET_GATE', 'CEC-'], init=init)
		
		add_add_layer(LAYERS, 'CEC', ['IN_MULT', 'CEC_MULT'], init=init)
		
		add_mult_layer(LAYERS, 'OUT', ['CEC', 'OUT_GATE'], init=init)
		
		## above
		add_sigmoid_F_bias_layer(LAYERS, 'A_F1', A_F1, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'A_F2', A_F2, init=init)
		add_linear_F_layer(LAYERS, 'SUM', 1, sum_all=True, init=init) # for each img, sum outputs
		
		add_add_layer(LAYERS, 'ERR', ['SUM', -1], scalar=-1, init=init)
		
		#####
		
		add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
		add_sum_layer(LAYERS, 'SUM_ERR', init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['CEC'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS
