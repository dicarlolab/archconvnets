from ntm_core import *

N_CTT = 3 # number of past frames to conv through time
N_IN = IM_SZ*IM_SZ*3
N_TARGET = N_IN

B = True # batch or not

A_F = 32 # number of units in above layers

def init_model():
	LAYERS = []

	U_F1_FILTER_SZ = 5
	U_F2_FILTER_SZ = 5
	U_F3_FILTER_SZ = 3
	
	U_F1 = 48*2
	U_F2 = 48*2
	U_F3 = 48*2
	
	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,N_CTT*3,IM_SZ,IM_SZ), PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		add_conv_layer(LAYERS, 'F2', U_F2, U_F2_FILTER_SZ, PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F2_MAX', init=init)
		add_conv_layer(LAYERS, 'F3', U_F3, U_F3_FILTER_SZ, PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F3_MAX', init=init)
		
		if CLASS_CIFAR:
			## cifar
			add_linear_F_bias_layer(LAYERS, 'CIFAR', 10, source='F3_MAX', batch_imgs=B, init=init)
			add_pearson_layer(LAYERS, 'CIFAR_ERR', ['CIFAR', -1], batch_imgs=B, init=init)
			add_sum_layer(LAYERS, 'CIFAR_SUM_ERR', init=init)
		
		## synthetic categorization
		add_linear_F_bias_layer(LAYERS, 'CAT', 10, source='F3_MAX', batch_imgs=B, init=init)
		add_pearson_layer(LAYERS, 'CAT_ERR', ['CAT', -1], batch_imgs=B, init=init)
		add_sum_layer(LAYERS, 'CAT_SUM_ERR', init=init)
		
		## synthetic identification
		add_linear_F_bias_layer(LAYERS, 'OBJ', 122, source='F3_MAX', batch_imgs=B, init=init)
		add_pearson_layer(LAYERS, 'OBJ_ERR', ['OBJ', -1], batch_imgs=B, init=init)
		add_sum_layer(LAYERS, 'OBJ_SUM_ERR', init=init)
		# bp_frame_pred uses prior layer as breakpoint for forward_network()
		
		if CLASS_IMGNET:
			## imgnet
			add_linear_F_bias_layer(LAYERS, 'IMGNET', 999, source='F3_MAX', batch_imgs=B, init=init)
			add_pearson_layer(LAYERS, 'IMGNET_ERR', ['IMGNET', -1], batch_imgs=B, init=init)
			add_sum_layer(LAYERS, 'IMGNET_SUM_ERR', init=init)
		# bp_frame_pred uses prior layer as breakpoint for forward_network()
		
		###################################
		## frame prediction
		add_sigmoid_F_bias_layer(LAYERS, 'M3_0', N_TARGET/2, source='F3_MAX', batch_imgs=B, init=init)
		
		# BUDGET = IM_SZ*IM_SZ*3 * U_F3*5*5
		# BUDGET / (48*32*32) = 75
		add_sigmoid_F_bias_layer(LAYERS, 'C1_0', 32, source='F1', batch_imgs=B, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'C1_1', N_TARGET/2, batch_imgs=B, init=init)
		
		add_add_layer(LAYERS, 'STACK_SUM0', ['C1_1', 'M3_0'], init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM1', N_TARGET, batch_imgs=B, init=init)
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM2', N_TARGET, batch_imgs=B, init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM2', -1], batch_imgs=B, init=init)
		add_sum_layer(LAYERS, 'SUM_ERR', init=init)
		
	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

