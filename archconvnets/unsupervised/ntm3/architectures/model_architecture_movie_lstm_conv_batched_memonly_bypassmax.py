from ntm_core import *

def init_model():
	LAYERS = []

	U_F1_FILTER_SZ = 5
	U_M_FILTER_SZ = 5
	
	U_F1 = 16
	U_M = 16
	
	N_TARGET = 16*16
	N_FC = 16*16

	for init in [0,1]:
		# below
		add_conv_layer(LAYERS, 'F1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,1,32,32), PAD=2, init=init)
		add_conv_layer(LAYERS, 'IN1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,1,32,32), PAD=2, init=init)
		
		add_conv_layer(LAYERS, 'IN_GATE_PRE1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,1,32,32), PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'IN_GATE1', init=init)
		
		add_conv_layer(LAYERS, 'FORGET_GATE_PRE1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,1,32,32), PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'FORGET_GATE1', init=init)
		
		add_conv_layer(LAYERS, 'OUT_GATE_PRE1', U_M, U_M_FILTER_SZ, source = -1, imgs_shape=(BATCH_SZ,1,32,32), PAD=2, init=init)
		add_sigmoid_layer(LAYERS, 'OUT_GATE1', init=init)
		
		
		add_mult_layer(LAYERS, 'IN_MULT1', ['IN1', 'IN_GATE1'], init=init)
		add_mult_layer(LAYERS, 'CEC_MULT1', ['FORGET_GATE1', 'CEC1-'], init=init)
		
		add_add_layer(LAYERS, 'CEC1', ['IN_MULT1', 'CEC_MULT1'], init=init)
		
		add_mult_layer(LAYERS, 'OUT1', ['CEC1', 'OUT_GATE1'], init=init)
		add_concat_layer(LAYERS, 'OUT12', ['OUT1', 'F1'], init=init)
		
		add_max_pool_layer(LAYERS, 'F1_MAX', init=init)
		
		#### lstm
		add_conv_layer(LAYERS, 'BYPASS', U_M, U_M_FILTER_SZ, source='F1_MAX', init=init)
		add_conv_layer(LAYERS, 'IN', U_M, U_M_FILTER_SZ, source='F1_MAX', init=init)
		
		add_conv_layer(LAYERS, 'IN_GATE_PRE', U_M, U_M_FILTER_SZ, source='F1_MAX', init=init)
		add_sigmoid_layer(LAYERS, 'IN_GATE', init=init)
		
		add_conv_layer(LAYERS, 'FORGET_GATE_PRE', U_M, U_M_FILTER_SZ, source='F1_MAX', init=init)
		add_sigmoid_layer(LAYERS, 'FORGET_GATE', init=init)
		
		add_conv_layer(LAYERS, 'OUT_GATE_PRE', U_M, U_M_FILTER_SZ, source='F1_MAX', init=init)
		add_sigmoid_layer(LAYERS, 'OUT_GATE', init=init)
		
		
		add_mult_layer(LAYERS, 'IN_MULT', ['IN', 'IN_GATE'], init=init)
		add_mult_layer(LAYERS, 'CEC_MULT', ['FORGET_GATE', 'CEC-'], init=init)
		
		add_add_layer(LAYERS, 'CEC', ['IN_MULT', 'CEC_MULT'], init=init)
		
		add_mult_layer(LAYERS, 'OUT', ['CEC', 'OUT_GATE'], init=init)
		add_concat_layer(LAYERS, 'OUT2', ['OUT', 'BYPASS'], init=init)
		
		add_max_pool_layer(LAYERS, 'CEC_MAX', init=init)
		
		add_sigmoid_F_bias_layer(LAYERS, 'A_F1', N_FC, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'MEM_STACK', N_FC, sum_all=True, init=init)
		
		### sum mem and conv stacks
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PRE3', N_FC, init=init)
		add_sigmoid_F_bias_layer(LAYERS, 'STACK_SUM_PRE4', N_FC, init=init)
		add_linear_F_bias_layer(LAYERS, 'STACK_SUM', N_TARGET, init=init)
		
		add_pearson_layer(LAYERS, 'ERR', ['STACK_SUM', -1], init=init)
		
		add_sum_layer(LAYERS,'ERR_SUM',init=init)

	check_network(LAYERS)
	
	################ init weights and inputs
	WEIGHTS = init_weights(LAYERS)
	MEM_INDS = find_layer(LAYERS, ['CEC1', 'CEC'])
	PX_INDS = find_layer(LAYERS, ['F1','IN1', 'IN_GATE_PRE1','FORGET_GATE_PRE1', 'OUT_GATE_PRE1'])
	PREV_VALS = random_function_list(LAYERS, MEM_INDS)
	
	return LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, PX_INDS

