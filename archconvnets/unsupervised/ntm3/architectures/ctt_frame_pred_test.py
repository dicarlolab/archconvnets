from ntm_core import *

BATCH_SZ = 100
N_CTT = 1 # number of past frames to conv through time
IM_SZ = 32
N_IN = IM_SZ*IM_SZ*3/4
N_TARGET = N_IN

B = True # batch or not

A_F = 32 # number of units in above layers

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
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,N_CTT*3,IM_SZ,IM_SZ), PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		add_conv_layer(LAYERS, 'F2', U_F2, U_F2_FILTER_SZ, PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F2_MAX', init=init)
		add_conv_layer(LAYERS, 'F3', U_F3, U_F3_FILTER_SZ, PAD=2, init=init)
		add_max_pool_layer(LAYERS, 'F3_MAX', init=init)
		
		###################################
		## frame prediction
		
		add_sigmoid_F_bias_layer(LAYERS, 'PX_0', N_TARGET, source=(BATCH_SZ,N_CTT*3,IM_SZ,IM_SZ), batch_imgs=B, init=init)
		#add_sigmoid_F_bias_layer(LAYERS, 'PX_1', N_TARGET, batch_imgs=B, init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'M3_0', N_TARGET, source='F3_MAX', batch_imgs=B, init=init)
		
		# BUDGET = IM_SZ*IM_SZ*3 * U_F3*5*5 = 32*32*3*48*5*5 = 
		# BUDGET / (48*32*32) = 75
		#add_sigmoid_F_bias_layer(LAYERS, 'M1_0', 512, source='F1_MAX', batch_imgs=B, init=init)
		#add_sigmoid_F_bias_layer(LAYERS, 'M1_1', N_TARGET, batch_imgs=B, init=init)
		
		#add_add_layer(LAYERS, 'STACK_SUM0', ['M1_1', 'M3_0'], init=init)
		add_add_layer(LAYERS, 'STACK_SUM1', ['M3_0', 'PX_0'], init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM2', N_TARGET, batch_imgs=B, init=init)
		add_linear_F_bias_layer(LAYERS, 'PRED', N_TARGET, batch_imgs=B, init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['PRED', -1], batch_imgs=B, init=init)
		add_sum_layer(LAYERS, 'SUM_ERR', init=init)
		
	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','F3']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

