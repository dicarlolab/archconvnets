import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import savemat
from scipy.stats import zscore, pearsonr
from model_architecture import init_model

free_all_buffers()

################ init save vars
save_name = 'ntm_test'

EPS = -1e-3
TIME_LENGTH = 3
elapsed_time = 1000
frame = 0
err = 0
n_saves = 0
training = 0
START_SIGNAL = 0; TRAIN_SIGNAL = 1
SAVE_FREQ = 25 # instantaneous checkpoint
WRITE_FREQ = 50 # new checkpoint
FRAME_LAG = 200
STOP_POINT = np.inf
inputs = np.zeros((2,1),dtype='single')

target_seq = np.abs(np.asarray(np.random.normal(size=TIME_LENGTH),dtype='single'))
output_seq = np.zeros_like(target_seq)

inputs[START_SIGNAL] = 1 # start signal

training_flag_buffer = np.zeros(SAVE_FREQ,dtype='single')
train_buffer = np.zeros(SAVE_FREQ,dtype='single')
target_buffer = np.zeros(SAVE_FREQ,dtype='single')
output_buffer = np.zeros(SAVE_FREQ,dtype='single')
err_log = []; corr_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()

ERR_IND = find_layer(LAYERS, 'ERR')
OUT_IND = find_layer(LAYERS, 'SUM')
F1_IND = find_layer(LAYERS, 'F1_lin')

OUTPUT = None; LOCAL_DERIVS = None; WEIGHT_DERIVS = None
MEM_DERIVS = [None]*len(MEM_INDS); WEIGHT_DERIVS_RMS = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

#####################
while True:
	inputs[START_SIGNAL] = 0
	
	# switch from training to testing or conversely
	if elapsed_time >= TIME_LENGTH:
		training = 1 - training
		elapsed_time = 0
		if training == 1: # new training sequence
			free_list(OUTPUT_PREV)
			free_partials(PARTIALS_PREV)
			
			OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
			PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

			target_seq = np.single(np.abs(np.random.normal(size=TIME_LENGTH)) + 2) #-.5
			inputs[START_SIGNAL] = 1
	
	if training == 1: # train period
		inputs[TRAIN_SIGNAL] = target_seq[elapsed_time]
		target = 0
	else: # test period
		inputs[TRAIN_SIGNAL] = 0
		target = target_seq[elapsed_time]

	###### forward
	set_buffer(inputs, WEIGHTS[F1_IND][1])  # inputs
	set_buffer(target, WEIGHTS[ERR_IND][1]) # target
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	if training != 1:
		err += return_buffer(OUTPUT[-1])
	
	training_flag_buffer[frame % SAVE_FREQ] = copy.deepcopy(training)
	train_buffer[frame % SAVE_FREQ] = copy.deepcopy(inputs[TRAIN_SIGNAL])
	target_buffer[frame % SAVE_FREQ] = copy.deepcopy(target)
	output_buffer[frame % SAVE_FREQ] =  return_buffer(OUTPUT[OUT_IND])
	
	###### reverse
	LOCAL_DERIVS = local_derivs(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, LOCAL_DERIVS)
	WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, WEIGHT_DERIVS)
	
	# update partials_prev
	MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, LOCAL_DERIVS, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
	PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
	
	OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	# take step
	if frame < STOP_POINT and frame > SAVE_FREQ:
		#update_weights(LAYERS, WEIGHTS, WEIGHT_DERIVS, EPS)
		WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS, frame, FRAME_LAG)
		
		
	# print
	if frame % SAVE_FREQ == 0 and frame != 0:
		corr_log.append(pearsonr(target_buffer[training_flag_buffer == 0], output_buffer[training_flag_buffer == 0])[0])
		err_log.append(err / SAVE_FREQ); err = 0
		
		print 'err: ', err_log[-1][0], 'frame: ', frame, 'corr: ', corr_log[-1], 'time: ', time.time() - t_start, save_name
		
		print_names = ['F1','F2','F3','', '_KEY', '_BETA', '_IN_GATE', '_SHIFT_PRE', '_GAMMA', '', 'ERASE', 'ADD', 'READ_MEM',\
			'A_F1', 'A_F2']
		
		max_print_len = 0
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
				print
		print '---------------------'
		
		#######
		
		#savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'target_buffer': target_buffer, 'err_log': err_log, 'corr_log': corr_log, 'EPS_BR': EPS_BR, 'EPS_WW': EPS_WW, 'EPS_WR': EPS_WR, 'EPS_BUNDER': EPS_BUNDER, 'EPS_WUNDER': EPS_WUNDER, 'EPS_WABOVE': EPS_WABOVE, 'EPS_BABOVE': EPS_BABOVE, 'training_flag_buffer': training_flag_buffer})
		
		t_start = time.time()
		
	'''# write
	if frame % WRITE_FREQ == 0:
		print 'writing', save_name
		#####
		
		file = open('/home/darren/ntm_saves/' + save_name + '_' + str(n_saves) + '.pk','w')
		pk.dump({'WR': WR, 'BR': BR, 'WW': WW, 'BW': BW, 'WUNDER': WUNDER, 'BUNDER': BUNDER, 'frame': frame, \
			'EPS_BABOVE': EPS_BABOVE, 'EPS_WABOVE': EPS_WABOVE, 'WABOVE': WABOVE, 'BABOVE': BABOVE,\
			'EPS_BR': EPS_BR, 'EPS_WW': EPS_WW, 'EPS_WR': EPS_WR, 'EPS_BUNDER': EPS_BUNDER, 'EPS_WUNDER': EPS_WUNDER}, file)
		file.close()
		
		n_saves += 1'''
		
	frame += 1
	elapsed_time += 1
	if frame == STOP_POINT:
		print 'stopping'
	if frame == (STOP_POINT + 3*SAVE_FREQ):
		break

free_list_list(MEM_DERIVS)
free_partials(PARTIALS_PREV)
free_list(LOCAL_DERIVS)
free_list(OUTPUT)
free_list(WEIGHT_DERIVS)
free_list(OUTPUT_PREV)
