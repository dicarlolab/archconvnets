from ntm_core import *

BATCH_SZ = 100
N_CTT = 3 # number of past frames to conv through time
IM_SZ = 32
N_IN = IM_SZ*IM_SZ*3
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
		add_linear_F_bias_layer(LAYERS, 'CIFAR', 32*32*3, source=(BATCH_SZ, 5,1), batch_imgs=B, init=init)
		add_pearson_layer(LAYERS, 'CIFAR_ERR', ['CIFAR', -1], batch_imgs=B, init=init)
		#add_add_layer(LAYERS, 'CIFAR_ERR', ['CIFAR', -1], init=init)
		add_sq_points_layer(LAYERS,'SQ',init=init)
		add_sum_layer(LAYERS, 'CIFAR_SUM_ERR', init=init)
		
	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','F3']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

