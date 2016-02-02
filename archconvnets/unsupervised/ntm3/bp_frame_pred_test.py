import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from architectures.ctt_frame_pred_test import *

N_MOVIES = 22872 #17750 #12340 #6372
BATCH_SZ = 50
EPS = 1e-2

DIFF = True
#DIFF = False

save_name = 'frame_pred_test_%f_%i' % (-EPS, N_MOVIES)

if DIFF:
	save_name += '_diff'

free_all_buffers()

################ init save vars

frame = 0; frame_local = 0; err = 0; corr = 0; movie_ind = 0; 
cifar_err = 0; cifar_class = 0
cat_err = 0; cat_class = 0; obj_err = 0; obj_class = 0

N_FUTURE = 1 # how far into the future to predict
EPOCH_LEN = 11 # length of movie
SAVE_FREQ = 25*BATCH_SZ #50 #250 # instantaneous checkpoint
WRITE_FREQ = 25*BATCH_SZ # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ

target_buffer = np.zeros((SAVE_FREQ, N_FRAMES_PRED*N_IN),dtype='single')
output_buffer = np.zeros((SAVE_FREQ, N_FRAMES_PRED*N_IN),dtype='single')

err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

STACK_SUM_IND = find_layer(LAYERS, 'STACK_SUM5') # network frame prediction
TARGET_IND = find_layer(LAYERS, 'ERR') # frame target

F1_IND = 0

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT = None; WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

WEIGHTS_F1_INIT = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])

#####################
while True:
	movie_frame = np.random.randint(EPOCH_LEN - N_CTT - N_FUTURE - N_FRAMES_PRED + 1) + N_CTT # movies
	
	# load movie
	movie_name = '/home/darren/rotating_objs32_constback_50t/imgs' + str(np.random.randint(N_MOVIES))  + '.mat'
	z = loadmat(movie_name)
	
	inputs = np.ascontiguousarray(np.single(z['imgs'] - .5))
	
	if DIFF:
		frame_target = (inputs[movie_frame] - inputs[movie_frame+N_FUTURE:movie_frame+N_FUTURE+N_FRAMES_PRED]).ravel()[:,np.newaxis]
	else:
		frame_target = inputs[movie_frame+N_FUTURE:movie_frame+N_FUTURE+N_FRAMES_PRED].ravel()[:,np.newaxis]
	frame_target = np.ascontiguousarray(frame_target)
	
	# load targets
	set_buffer(frame_target, WEIGHTS[TARGET_IND][1]) # frame target
	
	# forward movie
	set_buffer(inputs[movie_frame-N_CTT:movie_frame].reshape((1,N_CTT*3, IM_SZ, IM_SZ)), WEIGHTS[F1_IND][1])  # inputs
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	# predictions/errors
	time_series_prediction = return_buffer(OUTPUT[STACK_SUM_IND]).ravel()
	
	err += return_buffer(OUTPUT[TARGET_IND])
	
	output_buffer[frame % SAVE_FREQ] = copy.deepcopy(time_series_prediction)
	target_buffer[frame % SAVE_FREQ] = copy.deepcopy(frame_target.ravel())
	
	# reverse
	WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS, reset_derivs=False)
	
	if frame % BATCH_SZ == 0 and frame != 0:
		WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS / BATCH_SZ, frame, FRAME_LAG)
		
		zero_buffer_list(WEIGHT_DERIVS)
	
	# print/save
	if frame % SAVE_FREQ == 0 and frame != 0:
		corr_log.append(err / SAVE_FREQ); corr = 0
		err_log.append(err / SAVE_FREQ); err = 0
		cifar_err_log.append([cifar_err / SAVE_FREQ]); cifar_err = 0;
		cifar_class_log.append([np.single(cifar_class) / SAVE_FREQ]); cifar_class = 0;
		cat_err_log.append([cat_err / SAVE_FREQ]); cat_err = 0;
		cat_class_log.append([np.single(cat_class) / SAVE_FREQ]); cat_class = 0;
		obj_err_log.append([obj_err / SAVE_FREQ]); obj_err = 0;
		obj_class_log.append([np.single(obj_class) / SAVE_FREQ]); obj_class = 0;
		
		print_state(LAYERS, WEIGHTS, WEIGHT_DERIVS, OUTPUT, EPS, err_log, frame, corr_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names)
		
		t_start = time.time()
		
	frame += 1

free_list_list(MEM_DERIVS)
free_partials(PARTIALS_PREV)
free_list(OUTPUT)
free_list(WEIGHT_DERIVS)
free_list(OUTPUT_PREV)
