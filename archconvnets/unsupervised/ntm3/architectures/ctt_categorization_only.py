from ntm_core import *

BATCH_SZ = 100
N_CTT = 3 # number of past frames to conv through time
IM_SZ = 32
N_IN = IM_SZ*IM_SZ*3

B = True # batch or not

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
		
		## cifar
		add_linear_F_bias_layer(LAYERS, 'CIFAR', 10, source='F3_MAX', batch_imgs=B, init=init)
		add_pearson_layer(LAYERS, 'CIFAR_ERR', ['CIFAR', -1], batch_imgs=B, init=init)
		add_sum_layer(LAYERS, 'CIFAR_SUM_ERR', init=init)
		
		## synthetic categorization
		add_linear_F_bias_layer(LAYERS, 'CAT', 8, source='F3_MAX', batch_imgs=B, init=init)
		add_pearson_layer(LAYERS, 'CAT_ERR', ['CAT', -1], batch_imgs=B, init=init)
		add_sum_layer(LAYERS, 'CAT_SUM_ERR', init=init)
		
		## synthetic identification
		add_linear_F_bias_layer(LAYERS, 'OBJ', 32, source='F3_MAX', batch_imgs=B, init=init)
		add_pearson_layer(LAYERS, 'OBJ_ERR', ['OBJ', -1], batch_imgs=B, init=init)
		add_sum_layer(LAYERS, 'OBJ_SUM_ERR', init=init)
		
		## imgnet
		add_linear_F_bias_layer(LAYERS, 'IMGNET', 999, source='F3_MAX', batch_imgs=B, init=init)
		add_pearson_layer(LAYERS, 'IMGNET_ERR', ['IMGNET', -1], batch_imgs=B, init=init)
		add_sum_layer(LAYERS, 'IMGNET_SUM_ERR', init=init)
		
	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','F3']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

