import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from worlds.elastic_world import generate_latents

no_mem = True
no_mem = False

if no_mem:
	from architectures.movie_phys_latent_predict_series_no_mem import *
	INPUT_SCALE = 1e-5
	EPS = -1e-1
	save_name = 'ntm_physics_series_top_layers_no_mem_%f_n_pred_%i' % (-EPS, N_FRAMES_PRED)
else:
	from architectures.movie_phys_latent_predict_series import *
	INPUT_SCALE = 1e-5
	EPS = -1e-1
	save_name = 'ntm_physics_series_sm_mem_%f_n_pred_%i' % (-EPS, N_FRAMES_PRED)

	
free_all_buffers()


################ init save vars

frame = 0; frame_local = 0; err = 0; corr = 0

BATCH_SZ = 250
EPOCH_LEN = 6*6*2
SAVE_FREQ = 500 # instantaneous checkpoint
WRITE_FREQ = 50 # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ
STOP_POINT = np.inf #SAVE_FREQ*15

target_buffer = np.zeros((SAVE_FREQ, N_FRAMES_PRED*N_IN),dtype='single')
output_buffer = np.zeros((SAVE_FREQ, N_FRAMES_PRED*N_IN),dtype='single')
err_log = []; corr_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

STACK_SUM_IND = find_layer(LAYERS, 'STACK_SUM3')
TARGET_IND = find_layer(LAYERS, 'ERR')
OUT_IND = find_layer(LAYERS, 'ERR')
F1_IND = 0

OUTPUT = None; WEIGHT_DERIVS = None
MEM_DERIVS = [None]*len(MEM_INDS); WEIGHT_DERIVS_RMS = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

WEIGHTS_F1_INIT = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1_lin')][0])

#####################
while True:
	# end of movie:
	if (frame % EPOCH_LEN) == 0:
		#### new movie
		inputs, targets = generate_latents(EPOCH_LEN, N_FRAMES_PRED)
		
		#### reset state
		free_list(OUTPUT_PREV)
		free_partials(PARTIALS_PREV)
		
		OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
		PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
		
		frame_local = 0
	
	###### forward
	frame_target = targets[frame_local+1:frame_local+1+N_FRAMES_PRED].ravel()[:,np.newaxis]
	
	set_buffer(inputs[frame_local], WEIGHTS[F1_IND][1])  # inputs
	set_buffer(frame_target, WEIGHTS[TARGET_IND][1]) # target
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	time_series_prediction = return_buffer(OUTPUT[STACK_SUM_IND]).ravel()
	
	current_err = return_buffer(OUTPUT[-1])[0]
	err += current_err;
	corr += pearsonr(frame_target.ravel(), time_series_prediction)[0]

	output_buffer[frame % SAVE_FREQ] = copy.deepcopy(time_series_prediction)
	target_buffer[frame % SAVE_FREQ] = copy.deepcopy(frame_target.ravel())
	
	###### reverse
	WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS, reset_derivs=False)
	
	# update partials_prev
	MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
	PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
	
	OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	# take step
	if frame < STOP_POINT and frame != 0 and (frame % BATCH_SZ) == 0:
		WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS/BATCH_SZ, frame, FRAME_LAG)
		zero_buffer_list(WEIGHT_DERIVS)
		
		
	# print
	if frame % SAVE_FREQ == 0 and frame != 0:
		corr_log.append(corr / SAVE_FREQ); corr = 0
		err_log.append(err / SAVE_FREQ); err = 0
		
		print 'batch: ', frame, 'time: ', time.time() - t_start, 'GPU:', GPU_IND, save_name
		print 'err: ', err_log[-1], 'corr: ', corr_log[-1]
		print '------------'
		
		savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'target_buffer': target_buffer, \
				'err_log': err_log, 'corr_log': corr_log, 'EPS': EPS, 'EPOCH_LEN': EPOCH_LEN})
		
		t_start = time.time()
		
	frame += 1
	frame_local += 1
	if frame == STOP_POINT:
		print 'stopping'


free_list_list(MEM_DERIVS)
free_partials(PARTIALS_PREV)
free_list(OUTPUT)
free_list(WEIGHT_DERIVS)
free_list(OUTPUT_PREV)
