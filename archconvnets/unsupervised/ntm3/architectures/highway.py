from ntm_core import *

def init_model():
	LAYERS = []

	U_F1_FILTER_SZ = 3

	U_F1 = 3

	for init in [0,1]:
		add_conv_layer(LAYERS, 'F1_conv', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(1,3,6,6), init=init, PAD=1)
		
		add_bias_layer(LAYERS, 'F1_b', init=init)
		add_sigmoid_layer(LAYERS, 'F1', init=init)
		
		add_conv_layer(LAYERS, 'F1m', U_F1, U_F1_FILTER_SZ, source = 'STACK_SUM-', imgs_shape=(1,3,6,6), init=init, PAD=1)
		
		add_add_layer(LAYERS, 'STACK_SUM', ['F1','F1m'], init=init)
		
		add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
		add_sum_layer(LAYERS, 'SUM_ERR', init=init)

	check_network(LAYERS)

	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['STACK_SUM'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)

	print_names = ['F1','F1m', 'STACK_SUM']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names