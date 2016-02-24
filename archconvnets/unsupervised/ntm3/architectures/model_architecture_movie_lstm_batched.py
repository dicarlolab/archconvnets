from ntm_core import *

def init_model():
	LAYERS = []

	N_C = 32
	
	U_F1_FILTER_SZ = 5
	U_F2_FILTER_SZ = 5
	U_F3_FILTER_SZ = 3
	
	U_F1 = 16
	U_F2 = 16
	U_F3 = 16
	U_FL = 8
	
	N_TARGET = 16*16
	N_FC = 32*32*3*2
	HEAD_INPUT = 'F1_MAX'

	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_F1, U_F1_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,1,32,32), init=init)
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		
		#### lstm
		add_linear_F_bias_layer(LAYERS, 'IN', N_C, HEAD_INPUT, init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'IN_GATE', N_C, HEAD_INPUT, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'FORGET_GATE', N_C, HEAD_INPUT, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'OUT_GATE', N_C, HEAD_INPUT, init=init)
		
		add_mult_layer(LAYERS, 'IN_MULT', ['IN', 'IN_GATE'], init=init)
		add_mult_layer(LAYERS, 'CEC_MULT', ['FORGET_GATE', 'CEC-'], init=init)
		
		add_add_layer(LAYERS, 'CEC', ['IN_MULT', 'CEC_MULT'], init=init)
		
		add_mult_layer(LAYERS, 'OUT', ['CEC', 'OUT_GATE'], init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'A_F1', N_FC, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'MEM_STACK', N_FC, sum_all=True, init=init)
		
		### sum mem and conv stacks
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PX', N_FC, source=(BATCH_SZ,1,32,32), init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_F3', N_FC, source='F1_MAX', init=init)
		
		add_add_layer(LAYERS, 'STACK_SUM_PRE', ['STACK_SUM_F3', 'MEM_STACK'], init=init)
		add_add_layer(LAYERS, 'STACK_SUM_PRE2', ['STACK_SUM_PRE', 'STACK_SUM_PX'], init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PRE3', N_FC, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PRE4', N_FC, init=init)
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM', N_TARGET, init=init)
		
		#add_add_layer(LAYERS, 'ERR', ['STACK_SUM', -1], scalar=-1, init=init)
		#add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM', -1], init=init)
		
		add_sum_layer(LAYERS,'ERR_SUM',init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['CEC'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS

