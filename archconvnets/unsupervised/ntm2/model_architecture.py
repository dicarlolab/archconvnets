from ntm_core import *

def init_model():
	LAYERS = []

	N_CONTROLLERS = 16
	N_MEM_SLOTS = 6
	M_LENGTH = 8

	mem_shape = (N_MEM_SLOTS, M_LENGTH)

	N_F1 = 12
	N_F2 = 7
	N_F3 = 9
	HEAD_INPUT = 'F3'

	for init in [0,1]:
		# below
		add_linear_F_bias_layer(LAYERS, 'F1', N_F1, (2, 1), init=init)
		add_linear_F_bias_layer(LAYERS, 'F2', N_F2, init=init)
		add_linear_F_bias_layer(LAYERS, HEAD_INPUT, N_F3, init=init)
		
		for RW in ['R', 'W']:
			# content
			add_linear_F_bias_layer(LAYERS, RW+'_KEY', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
			add_cosine_sim_layer(LAYERS, RW+'_CONTENT', [RW+'_KEY', 'MEM-'], mem_shape, init=init)
			add_linear_F_bias_layer(LAYERS, RW+'_BETA', N_CONTROLLERS, HEAD_INPUT, init=init)
			add_focus_keys_layer(LAYERS, RW+'_CONTENT_FOCUSED', [RW+'_CONTENT', RW+'_BETA'], init=init)
			add_softmax_layer(LAYERS, RW+'_CONTENT_SM', init=init)
			
			# interpolate
			add_linear_F_bias_layer(LAYERS, RW+'_IN_GATE_PRE', N_CONTROLLERS, HEAD_INPUT, init=init)
			add_sigmoid_layer(LAYERS, RW+'_IN_GATE', init=init)
			add_interpolate_layer(LAYERS, RW+'_IN_PRE', [RW+'_IN_GATE', RW+'_CONTENT_SM', RW+'_F-'], init=init)
			add_softmax_layer(LAYERS, RW+'_IN', init=init)
			
			# shift
			add_linear_F_bias_layer(LAYERS, RW+'_SHIFT_PRE', (N_CONTROLLERS, N_SHIFTS), HEAD_INPUT, init=init)
			add_softmax_layer(LAYERS, RW+'_SHIFT', init=init)
			add_shift_w_layer(LAYERS, RW+'_SHIFTED', [RW+'_SHIFT', RW+'_IN'], init=init)
			
			# sharpen
			add_linear_F_bias_layer(LAYERS, RW+'_GAMMA_PRE', N_CONTROLLERS, HEAD_INPUT, init=init)
			add_relu_layer(LAYERS, RW+'_GAMMA', init=init)
			add_sharpen_layer(LAYERS, RW+'_F', [RW+'_SHIFTED', RW+'_GAMMA'], init=init)
		
		##### write
		# erase/add output
		add_linear_F_bias_layer(LAYERS, 'ERASE_PRE', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
		add_sigmoid_layer(LAYERS, 'ERASE', init=init)
		
		add_linear_F_bias_layer(LAYERS, 'ADD_PRE', (N_CONTROLLERS, M_LENGTH), HEAD_INPUT, init=init)
		add_sigmoid_layer(LAYERS, 'ADD', init=init)
		
		add_dotT_layer(LAYERS, 'ERASE_HEAD', ['W_F', 'ERASE'], init=init)
		add_dotT_layer(LAYERS, 'ADD_HEAD', ['W_F', 'ADD'], init=init)
		
		# mem = mem_prev * (1 - dotT(W_F, ERASE) + dotT(W_F, ADD)
		add_mult_layer(LAYERS, 'MEM_ERASE', ['ERASE_HEAD', 'MEM-'], init=init)
		add_add_layer(LAYERS, 'MEM_ERASED', ['MEM-', 'MEM_ERASE'], scalar=-1, init=init)
		add_add_layer(LAYERS, 'MEM', ['ADD_HEAD', 'MEM_ERASED'], init=init)
		
		##### read
		add_linear_F_bias_layer(LAYERS, 'READ_MEM', N_CONTROLLERS, 'R_F', 'MEM-', init=init)
		
		add_sq_points_layer(LAYERS, 'SQ', init=init)
		add_sum_layer(LAYERS, 'SUM', init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['MEM', 'R_F', 'W_F'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS