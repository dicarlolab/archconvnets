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
		
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM5', N_TARGET, source='F1',init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM5', -1], init=init)
		
		
	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

