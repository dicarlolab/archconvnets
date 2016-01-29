from ntm_core import *

N_FRAMES_PRED = 1
N_CTT = 3 # number of past frames to conv through time
IM_SZ = 32
N_IN = IM_SZ*IM_SZ*3

def init_model():
	LAYERS = []

	U_F1_FILTER_SZ = 5
	U_F2_FILTER_SZ = 5
	U_F3_FILTER_SZ = 3
	
	U_F1 = 48
	U_F2 = 48
	U_F3 = 48
	
	A_F0 = 256
	A_F1 = 256
	
	N_TARGET = N_IN*N_FRAMES_PRED
	
	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(1,N_CTT*3,IM_SZ,IM_SZ), PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		add_conv_layer(LAYERS, 'F2', U_F2, U_F2_FILTER_SZ, PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F2_MAX', init=init)
		add_conv_layer(LAYERS, 'F3', U_F3, U_F3_FILTER_SZ, PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F3_MAX', init=init)
		
		## cifar
		add_linear_F_bias_layer(LAYERS, 'CIFAR', 10, source='F3_MAX', init=init)
		add_pearson_layer(LAYERS, 'CIFAR_ERR', ['CIFAR', -1], init=init)
		
		## synthetic categorization
		add_linear_F_bias_layer(LAYERS, 'SYN_CAT', 8, source='F3_MAX', init=init)
		add_pearson_layer(LAYERS, 'SYN_CAT_ERR', ['SYN_CAT', -1], init=init)
		
		## synthetic identification
		add_linear_F_bias_layer(LAYERS, 'SYN_OBJ', 32, source='F3_MAX', init=init)
		add_pearson_layer(LAYERS, 'SYN_OBJ_ERR', ['SYN_OBJ', -1], init=init)
		
		###################################
		## frame prediction
		
		## above inputs (Max3)
		add_sigmoid_F_bias_layer(LAYERS, 'M3_F0', A_F0, source='F3_MAX', init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'M3_F1', A_F1, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'M3_F2', N_TARGET, init=init)
		
		## above inputs (conv1)
		add_sigmoid_F_bias_layer(LAYERS, 'C1_F0', A_F0, source='F1', init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'C1_F1', A_F1, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'C1_F2', N_TARGET, init=init)
		
		## above inputs (max2)
		add_sigmoid_F_bias_layer(LAYERS, 'M2_F0', A_F0, source='F2_MAX', init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'M2_F1', A_F1, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'M2_F2', N_TARGET, init=init)
		
		add_add_layer(LAYERS, 'STACK_SUM2', ['M3_F2', 'C1_F2'], init=init)
		add_add_layer(LAYERS, 'STACK_SUM3', ['STACK_SUM2', 'M2_F2'], init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM4', N_TARGET, init=init)
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM5', N_TARGET, init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM5', -1], init=init)
		
	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','F3']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

