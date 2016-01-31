from ntm_core import *

N_CTT = 1 # number of past frames to conv through time
IM_SZ = 32

N_C = 3 # game directions M, L, R

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
		
		## game
		# each action gets its own prediction
		# so we can backprop errors for individual actions
		for action in range(N_C):
			add_linear_F_bias_layer(LAYERS, 'GAME_PRED_'+str(action), 1, source='F3_MAX', init=init)
			add_add_layer(LAYERS, 'SUM_'+str(action), ['GAME_PRED_'+str(action), -1], scalar=-1, init=init)
			add_sq_points_layer(LAYERS, 'SQ_ERR_'+str(action), init=init)
			add_sum_layer(LAYERS, 'SUM_ERR_'+str(action), init=init)
		
	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = []
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	print_names = ['F1','F2','F3']
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names

def print_reinforcement_state(LAYERS, WEIGHTS, WEIGHT_DERIVS, OUTPUT, EPS, CHANCE_RAND, err_log, frame, r_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names):
	print 'err: ', err_log[-1][0], 'r:', r_log[-1][0], 'frame: ', frame, 'cifar_class: ', cifar_class_log[-1][0], 'time: ', time.time() - t_start, 'GPU:', GPU_IND, save_name
	print 'obj_class: ', obj_class_log[-1][0], 'cat_class: ', cat_class_log[-1][0], 'chance_rand:', CHANCE_RAND
	
	'''max_print_len = 0
	for print_name in print_names:
		if len(print_name) > max_print_len:
			max_print_len = len(print_name)
	
	for print_name in print_names:
		if len(print_name) != 0: # print layer
			if print_name[0] != '_': # standard layer
				print_layer(LAYERS, print_name, WEIGHTS, WEIGHT_DERIVS, OUTPUT, max_print_len, EPS)
			else: # read/write layers
				print_layer(LAYERS, 'R'+print_name, WEIGHTS, WEIGHT_DERIVS, OUTPUT, max_print_len, EPS)
				print_layer(LAYERS, 'W'+print_name, WEIGHTS, WEIGHT_DERIVS, OUTPUT, max_print_len, EPS)
		else: # print blank
			print'''
	print '---------------------'
