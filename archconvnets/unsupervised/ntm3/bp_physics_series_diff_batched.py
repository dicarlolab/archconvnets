import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from worlds.elastic_world_batched import generate_imgs

train = 5
EPS = 1e-2

if train == 0:
	from architectures.model_architecture_movie_no_mem_batched import init_model
	save_name = 'ntm_physics_series_diff_no_mem2_%f' % EPS
elif train == 1:
	from architectures.model_architecture_movie_mem_batched import init_model
	save_name = 'ntm_physics_series_diff_%f' % EPS
elif train == 2:
	from architectures.model_architecture_movie_lstm_batched import init_model
	save_name = 'lstm_physics_series_diff_%f' % EPS
elif train == 3:
	from architectures.model_architecture_movie_lstm_conv_batched import init_model
	save_name = 'lstm_conv_physics_series_diff_%f' % EPS
elif train == 4:
	from architectures.model_architecture_movie_lstm_conv_stacked_batched import init_model
	save_name = 'lstm_conv_stacked_physics_series_diff_%f' % EPS
elif train == 5:
	from architectures.model_architecture_movie_lstm_conv_stacked_batched import init_model
	save_name = 'lstm_conv_stacked_physics_series_diff_no_mem_%f' % EPS

	
free_all_buffers()

################ init save vars
TIME_LENGTH = 3
EPOCH_LEN = TIME_LENGTH*2
SAVE_FREQ = EPOCH_LEN*2 # instantaneous checkpoint
FRAME_LAG = 50
STOP_POINT = np.inf
INPUT_SCALE = 1e-1

frame = 0; err = 0; frame_local = EPOCH_LEN; frame_save = 0

input_buffer = np.zeros((SAVE_FREQ, BATCH_SZ, 1, 32, 32), dtype='single')
output_buffer = np.zeros((SAVE_FREQ, BATCH_SZ, 16*16, 1), dtype='single')
target_buffer = np.zeros((SAVE_FREQ, BATCH_SZ, 16*16, 1), dtype='single')
err_t_series = np.zeros(EPOCH_LEN, dtype='single')

err_log = []; err_t_series_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()

STACK_SUM_PX_IND = find_layer(LAYERS, 'STACK_SUM_PX_lin')
STACK_SUM_IND = find_layer(LAYERS, 'STACK_SUM')
TARGET_IND = find_layer(LAYERS, 'ERR')
F1_IND = 0

OUTPUT = None; WEIGHT_DERIVS = None
MEM_DERIVS = [None]*len(MEM_INDS); WEIGHT_DERIVS_RMS = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

WEIGHTS_F1_INIT = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])

#####################
while True:
	# end of movie:
	if frame_local == EPOCH_LEN:
		#### new movie
		inputs, targets = generate_imgs(EPOCH_LEN)
		
		#### reset state
		free_list(OUTPUT_PREV)
		free_partials(PARTIALS_PREV)
		
		OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
		PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
		
		frame_local = 0
	
	###### forward
	if frame_local < TIME_LENGTH: # load inputs
		set_buffer(inputs[frame_local], WEIGHTS[F1_IND][1])  # inputs
		set_buffer(inputs[frame_local], WEIGHTS[STACK_SUM_PX_IND][1])  # inputs
		t = np.zeros_like(targets[frame_local])
		
		input_buffer[frame_save] = copy.deepcopy(inputs[frame_local])
		target_buffer[frame_save] = copy.deepcopy(targets[frame_local])
	else: # predict future
		t = targets[frame_local] - targets[TIME_LENGTH - 2]
	
		input_buffer[frame_save] = copy.deepcopy(inputs[TIME_LENGTH - 2])
	
	t +=  np.random.normal(scale=1e-3, size=t.shape)
	set_buffer(t, WEIGHTS[TARGET_IND][1]) # target
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	target_buffer[frame_save] = copy.deepcopy(t)
	output_buffer[frame_save] = return_buffer(OUTPUT[STACK_SUM_IND])
	
	###### reverse
	if frame_local >= TIME_LENGTH:
		WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
		
		current_err = return_buffer(OUTPUT[-1])[0]
		err += current_err
		err_t_series[frame_local] += current_err
	
	# update partials_prev
	MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
	PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
	
	OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	# take step
	if frame < STOP_POINT and frame_local >= TIME_LENGTH:
		WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS / BATCH_SZ, frame, FRAME_LAG)
		
		if frame == FRAME_LAG:
			print 'starting'
		
	# print
	if (frame_save+1) % SAVE_FREQ == 0 and frame != 0:
		frame_save = -1
		err_t_series_log.append(err_t_series / (BATCH_SZ*SAVE_FREQ/EPOCH_LEN))
		err_t_series = np.zeros(EPOCH_LEN, dtype='single')
		
		err_log.append(err / (BATCH_SZ*SAVE_FREQ)); err = 0
		
		print 'err: ', err_log[-1], 'batch: ', frame, 'time: ', time.time() - t_start, 'GPU:', GPU_IND, 'batch_sz:', BATCH_SZ, save_name
		print 'err_t_series: ', err_t_series_log[-1][TIME_LENGTH:]
		
		#######
		WEIGHTS_F1 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])
		savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'err_log': err_log, 'EPS': EPS, \
				'F1_init': WEIGHTS_F1_INIT, 'F1': WEIGHTS_F1, 'EPOCH_LEN': EPOCH_LEN, 'err_t_series': err_t_series_log, 'BATCH_SZ': BATCH_SZ,
				'input_buffer': input_buffer, 'target_buffer': target_buffer, 'output_buffer': output_buffer})
		
		t_start = time.time()
		
	frame += 1; frame_local += 1; frame_save += 1
	if frame == STOP_POINT:
		print 'stopping'
