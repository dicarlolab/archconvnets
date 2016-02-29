from ntm_core import *

HEAD_INPUT = 'CEC2_MAX'

def init_model():
	LAYERS = []

	U_F1_FILTER_SZ = 5
	U_M_FILTER_SZ = 5
	
	IMG_SHAPE = (BATCH_SZ,3,32,32)
	
	U_F1 = 32
	U_M = 32
	
	N_FC = 3*16*16

	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=IMG_SHAPE, PAD=2, init=init)
		add_conv_layer(LAYERS, 'IN1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=IMG_SHAPE, PAD=2, init=init)
		
		add_conv_layer(LAYERS, 'IN_GATE_PRE1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=IMG_SHAPE, PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'IN_GATE1', init=init)
		
		add_conv_layer(LAYERS, 'FORGET_GATE_PRE1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=IMG_SHAPE, PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'FORGET_GATE1', init=init)
		
		add_conv_layer(LAYERS, 'OUT_GATE_PRE1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=IMG_SHAPE, PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'OUT_GATE1', init=init)
		
		
		add_mult_layer(LAYERS, 'IN_MULT1', ['IN1', 'IN_GATE1'], init=init)
		add_mult_layer(LAYERS, 'CEC_MULT1', ['FORGET_GATE1', 'CEC1-'], init=init)
		
		add_add_layer(LAYERS, 'CEC1', ['IN_MULT1', 'CEC_MULT1'], init=init)
		
		add_mult_layer(LAYERS, 'OUT1', ['CEC1', 'OUT_GATE1'], init=init)
		add_add_layer(LAYERS, 'OUT2', ['OUT1', 'F1'], init=init)
		
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		
		#### lstm
		add_conv_layer(LAYERS, 'BYPASS', U_M, U_M_FILTER_SZ, source='F1_MAX', PAD=2, init=init)
		add_conv_layer(LAYERS, 'IN', U_M, U_M_FILTER_SZ, source='F1_MAX', PAD=2, init=init)
		
		add_conv_layer(LAYERS, 'IN_GATE_PRE', U_M, U_M_FILTER_SZ, source='F1_MAX', PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'IN_GATE', init=init)
		
		add_conv_layer(LAYERS, 'FORGET_GATE_PRE', U_M, U_M_FILTER_SZ, source='F1_MAX', PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'FORGET_GATE', init=init)
		
		add_conv_layer(LAYERS, 'OUT_GATE_PRE', U_M, U_M_FILTER_SZ, source='F1_MAX', PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'OUT_GATE', init=init)
		
		
		add_mult_layer(LAYERS, 'IN_MULT', ['IN', 'IN_GATE'], init=init)
		add_mult_layer(LAYERS, 'CEC_MULT', ['FORGET_GATE', 'CEC-'], init=init)
		
		add_add_layer(LAYERS, 'CEC', ['IN_MULT', 'CEC_MULT'], init=init)
		
		add_mult_layer(LAYERS, 'OUT', ['CEC', 'OUT_GATE'], init=init)
		add_add_layer(LAYERS, 'OUT0', ['OUT', 'BYPASS'], init=init)
		
		add_max_pool_layer(LAYERS, 'CEC_MAX', init=init)
		
		#### lstm
		add_conv_layer(LAYERS, 'BYPASS3', U_M, U_M_FILTER_SZ, source='OUT2', PAD=2, init=init)
		add_conv_layer(LAYERS, 'IN3', U_M, U_M_FILTER_SZ, source='OUT2', PAD=2, init=init)
		
		add_conv_layer(LAYERS, 'IN_GATE_PRE3', U_M, U_M_FILTER_SZ, source='OUT2', PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'IN_GATE3', init=init)
		
		add_conv_layer(LAYERS, 'FORGET_GATE_PRE3', U_M, U_M_FILTER_SZ, source='OUT2', PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'FORGET_GATE3', init=init)
		
		add_conv_layer(LAYERS, 'OUT_GATE_PRE3', U_M, U_M_FILTER_SZ, source='OUT2', PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'OUT_GATE3', init=init)
		
		
		add_mult_layer(LAYERS, 'IN_MULT3', ['IN3', 'IN_GATE3'], init=init)
		add_mult_layer(LAYERS, 'CEC_MULT3', ['FORGET_GATE3', 'CEC3-'], init=init)
		
		add_add_layer(LAYERS, 'CEC3', ['IN_MULT3', 'CEC_MULT3'], init=init)
		
		add_mult_layer(LAYERS, 'OUT3', ['CEC3', 'OUT_GATE3'], init=init)
		add_add_layer(LAYERS, 'OUT4', ['OUT3', 'BYPASS3'], init=init)
		
		add_max_pool_layer(LAYERS, 'CEC2_MAX', init=init)
		
		## synthetic categorization
		
		add_linear_F_bias_layer(LAYERS, 'CAT', 7, source='CEC2_MAX', init=init)
		#add_linear_F_bias_layer(LAYERS, 'STACK_SUM1_CAT', 7, source='CEC_MAX', init=init)
		#add_linear_F_bias_layer(LAYERS, 'STACK_SUM2_CAT', 7, source='F1_MAX', init=init)
		#add_add_layer(LAYERS, 'STACK_SUM3_CAT', ['STACK_SUM0_CAT', 'STACK_SUM1_CAT'], init=init)
		#add_add_layer(LAYERS, 'CAT', ['STACK_SUM3_CAT', 'STACK_SUM2_CAT'], init=init)
		
		add_pearson_layer(LAYERS, 'CAT_ERR', ['CAT', -1], init=init)
		add_sum_layer(LAYERS, 'CAT_SUM_ERR', init=init)
		
		## synthetic identification
		
		add_linear_F_bias_layer(LAYERS, 'OBJ', 91, source='CEC2_MAX', init=init)
		#add_linear_F_bias_layer(LAYERS, 'STACK_SUM1_OBJ', 91, source='CEC_MAX', init=init)
		#add_linear_F_bias_layer(LAYERS, 'STACK_SUM2_OBJ', 91, source='F1_MAX', init=init)
		#add_add_layer(LAYERS, 'STACK_SUM3_OBJ', ['STACK_SUM0_OBJ', 'STACK_SUM1_OBJ'], init=init)
		#add_add_layer(LAYERS, 'OBJ', ['STACK_SUM3_OBJ', 'STACK_SUM2_OBJ'], init=init)
		
		add_pearson_layer(LAYERS, 'OBJ_ERR', ['OBJ', -1], init=init)
		add_sum_layer(LAYERS, 'OBJ_SUM_ERR', init=init)
		# bp_frame_wise_pred_seq uses prior layer as breakpoint for forward_network()
		
		### sum mem and conv stacks
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM0', N_TARGET, source='CEC2_MAX', init=init)
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM1', N_TARGET, source='CEC_MAX', init=init)
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM2', N_TARGET, source='F1_MAX', init=init)
		add_add_layer(LAYERS, 'STACK_SUM3', ['STACK_SUM0', 'STACK_SUM1'], init=init)
		add_add_layer(LAYERS, 'STACK_SUM4', ['STACK_SUM3', 'STACK_SUM2'], init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM4', -1], init=init)
		
		add_sum_layer(LAYERS,'ERR_SUM',init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	
	MEM_INDS = find_layer(LAYERS, ['CEC1', 'CEC', 'CEC3'])
	PX_INDS = find_layer(LAYERS, ['F1','IN1', 'IN_GATE_PRE1','FORGET_GATE_PRE1', 'OUT_GATE_PRE1'])
	
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, PX_INDS

