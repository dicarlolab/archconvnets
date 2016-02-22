import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import savemat
from scipy.stats import zscore, pearsonr

no_mem = True
no_mem = False

EPS = -1e-1

if no_mem:
	from architectures.model_architecture_cp_no_mem import init_model
	save_name = 'ntm_no_mem_%f' % (-EPS)
else:
	from architectures.model_architecture_cp_batched import init_model
	save_name = 'ntm_cp_uniform_batch_mem_%f' % (-EPS)

free_all_buffers()

################ init save vars
TIME_LENGTH = 3
elapsed_time = np.inf
frame = 0
err = 0
n_saves = 0
training = 0
START_SIGNAL = 0; TRAIN_SIGNAL = 1
SAVE_FREQ = 25 # instantaneous checkpoint
FRAME_LAG = 5
STOP_POINT = np.inf #5000 #np.inf #SAVE_FREQ*15
inputs = np.zeros((BATCH_SZ, 2,1),dtype='single')

target_seq = np.abs(np.asarray(np.random.normal(size=(BATCH_SZ, TIME_LENGTH,1)),dtype='single'))
output_seq = np.zeros_like(target_seq)

inputs[:,START_SIGNAL] = 1 # start signal

training_flag_buffer = np.zeros(SAVE_FREQ, dtype='single')
train_buffer = np.zeros((SAVE_FREQ, BATCH_SZ, 1), dtype='single')
target_buffer = np.zeros((SAVE_FREQ, BATCH_SZ, 1), dtype='single')
output_buffer = np.zeros((SAVE_FREQ, BATCH_SZ, 1), dtype='single')
err_log = []; corr_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()

ERR_IND = find_layer(LAYERS, 'ERR')
OUT_IND = find_layer(LAYERS, 'SUM')
F1_IND = find_layer(LAYERS, 'F1_lin')

OUTPUT = None; WEIGHT_DERIVS = None
MEM_DERIVS = [None]*len(MEM_INDS); WEIGHT_DERIVS_RMS = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

#####################
while True:
	inputs[:,START_SIGNAL] = 0
	
	# switch from training to testing or conversely
	if elapsed_time >= TIME_LENGTH:
		training = 1 - training
		elapsed_time = 0
		if training == 1: # new training sequence
			free_list(OUTPUT_PREV)
			free_partials(PARTIALS_PREV)
			
			OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
			PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
			
			target_seq = np.single(np.abs(np.random.normal(size=(BATCH_SZ, TIME_LENGTH,1))) + 2) #-.5
			inputs[:,START_SIGNAL] = 1
	
	if training == 1: # train period
		inputs[:,TRAIN_SIGNAL] = target_seq[:,elapsed_time]
		target = np.zeros((BATCH_SZ, 1), dtype='single')
	else: # test period
		inputs[:,TRAIN_SIGNAL] = 0
		target = target_seq[:,elapsed_time]

	###### forward
	set_buffer(np.ascontiguousarray(inputs), WEIGHTS[F1_IND][1])  # inputs
	set_buffer(np.ascontiguousarray(target), WEIGHTS[ERR_IND][1]) # target
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	if training != 1:
		err += return_buffer(OUTPUT[-1])[0]
	
	training_flag_buffer[frame % SAVE_FREQ] = copy.deepcopy(training)
	train_buffer[frame % SAVE_FREQ] = copy.deepcopy(inputs[:,TRAIN_SIGNAL])
	target_buffer[frame % SAVE_FREQ] = copy.deepcopy(target)
	output_buffer[frame % SAVE_FREQ] =  return_buffer(OUTPUT[OUT_IND])#[:,np.newaxis]
	
	###### reverse
	WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
	
	# update partials_prev
	MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
	PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
	
	OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	# take step
	if frame < STOP_POINT:
		WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS/BATCH_SZ, frame, FRAME_LAG)
		#update_weights(LAYERS, WEIGHTS, WEIGHT_DERIVS, EPS/BATCH_SZ)
		
	# print
	if frame % SAVE_FREQ == 0 and frame != 0:
		corr_log.append(pearsonr(target_buffer[training_flag_buffer == 0].ravel(), output_buffer[training_flag_buffer == 0].ravel())[0])
		err_log.append(err / (BATCH_SZ*SAVE_FREQ)); err = 0
		
		print 'batch: ', frame, 'time: ', time.time() - t_start, 'GPU:', GPU_IND, save_name
		print 'err: ', err_log[-1], 'corr: ', corr_log[-1]
		print '------------'
		
		#######
		
		savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'target_buffer': target_buffer, 'err_log': err_log, 'corr_log': corr_log, 'EPS': EPS, 'training_flag_buffer': training_flag_buffer})
		
		t_start = time.time()
		
	frame += 1
	elapsed_time += 1
	if frame == STOP_POINT:
		print 'stopping'
