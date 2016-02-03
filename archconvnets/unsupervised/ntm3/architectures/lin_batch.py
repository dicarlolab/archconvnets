from ntm_core import *

IM_SZ = 32
BATCH_SZ = 5

def init_model():
	LAYERS = []

	U_F1_FILTER_SZ = 5
	U_F2_FILTER_SZ = 5
	U_F3_FILTER_SZ = 3
	
	U_F1 = 48
	U_F2 = 48
	U_F3 = 48
	
	for init in [0,1]:
		# below
		add_linear_F_layer(LAYERS, 'CIFAR', 10, source=(5,3,4), batch_imgs=True, init=init)
		add_linear_F_layer(LAYERS, 'CIFAR2', 20, batch_imgs=True, init=init)
		
		#add_conv_layer(LAYERS, 'F2', U_F1, U_F1_FILTER_SZ, PAD=2, init=init)
		#add_max_pool_layer(LAYERS, 'F2_MAX', init=init)
		#add_conv_layer(LAYERS, 'F3', U_F1, U_F1_FILTER_SZ, PAD=2, init=init)
		#add_max_pool_layer(LAYERS, 'F3_MAX', init=init)
		'''#add_relu_layer(LAYERS, 'F1_relu', init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		add_conv_layer(LAYERS, 'F2', U_F2, U_F2_FILTER_SZ, PAD=2, init=init)
		#add_relu_layer(LAYERS, 'F2_relu', init=init)
		add_max_pool_layer(LAYERS, 'F2_MAX', init=init)
		add_conv_layer(LAYERS, 'F3', U_F3, U_F3_FILTER_SZ, PAD=2, init=init)
		#add_relu_layer(LAYERS, 'F3_relu', init=init)
		add_max_pool_layer(LAYERS, 'F3_MAX', init=init)
		
		## cifar
		#add_linear_F_bias_layer(LAYERS, 'CIFAR', 10, source='F3_MAX', init=init)
		#add_pearson_layer(LAYERS, 'CIFAR_ERR', ['CIFAR', -1], init=init)
		'''
		#add_sq_points_layer(LAYERS,'SQ',init=init)
		add_sum_layer(LAYERS, 'SUM', init=init)
		
		
	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','F3']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

