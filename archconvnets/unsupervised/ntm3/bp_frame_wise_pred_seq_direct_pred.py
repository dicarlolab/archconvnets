import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
#from architectures.model_architecture_movie_lstm_conv_framewise import *
#from architectures.model_architecture_movie_lstm_conv_framewise_bypassmax_minFC_npool import *
#from architectures.movie_lstm_pool import *
from architectures.movie_lstm_1layer_direct import *
from img_sets.movie_seqs_framewise import *

EPS = 7.5e-2

train_filters_on = 0

free_all_buffers()
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, PX_INDS = init_model()
abort_cat = abort_obj = HEAD_INPUT
MOVIE_BREAK_LAYER_IND = find_layer(LAYERS, 'OBJ_SUM_ERR')

if train_filters_on == 0:
	save_name = 'frame_pred'
	MOVIE_BREAK_LAYER_IND = None
elif train_filters_on == 1:
	save_name = 'cat'
	abort_cat = None
elif train_filters_on == 2:
	save_name = 'obj'
	abort_obj = None
else:
	save_name = 'rand'
 
save_name += '_EPS_%f_lstm_1layer_direct_N_FUTURE_%i' % (EPS, N_FUTURE)

if NO_MEM:
	save_name += '_no_mem'

################ init save vars
batch = 0; err = 0; err_test = 0; movie_ind = 0; 
cat_err = 0; cat_class = 0; cat_test_err = 0; cat_test_class = 0; 
obj_err = 0; obj_class = 0; obj_test_err = 0; obj_test_class = 0

SAVE_FREQ = 20
batch_LAG = 10
batch_stop = np.inf #150*2

err_log = []; err_t_series_log = []; err_t_series_test_log = []; err_test_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []
cat_test_err_log = []; cat_test_class_log = []; obj_test_err_log = []; obj_test_class_log = []

output_buffer = np.zeros((EPOCH_LEN/2, BATCH_SZ, N_TARGET, 1), dtype='single')
output_buffer_test = np.zeros((EPOCH_LEN/2, BATCH_SZ, N_TARGET, 1), dtype='single')
err_t_series = np.zeros(EPOCH_LEN/2, dtype='single')
err_t_series_test = np.zeros(EPOCH_LEN/2, dtype='single')

################ init weights and inputs
PRED_IND = find_layer(LAYERS, 'STACK_SUM4')

OBJ_PRED_IND = find_layer(LAYERS, 'OBJ')
CAT_PRED_IND = find_layer(LAYERS, 'CAT')

DIFF_IND = find_layer(LAYERS, 'ERR')
CAT_DIFF_IND = find_layer(LAYERS, 'CAT_ERR')
OBJ_DIFF_IND = find_layer(LAYERS, 'OBJ_ERR')

OUT_IND = find_layer(LAYERS, 'ERR_SUM')
OBJ_OUT_IND = find_layer(LAYERS, 'OBJ_SUM_ERR')
CAT_OUT_IND = find_layer(LAYERS, 'CAT_SUM_ERR')

F1_IND = 0

OUTPUT = [None]*(1+EPOCH_LEN); WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None
OUTPUT_OBJ = None; WEIGHT_DERIVS_OBJ = None; WEIGHT_DERIVS_RMS_OBJ = None
OUTPUT_CAT = None; WEIGHT_DERIVS_CAT = None; WEIGHT_DERIVS_RMS_CAT = None
OUTPUT[0] = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)

t_start = time.time()

#####################
while True:
	###############
	# forward movie
	
	# go through the frames:
	for frame in range(3):
		# new frame
		objs, cats, cat_target, obj_target, movie_inputs, frame_target = load_movie_seqs(batch, frame, CAT_DIFF_IND, OBJ_DIFF_IND, DIFF_IND, PX_INDS, WEIGHTS)
		OUTPUT[frame+1] = forward_network(LAYERS, WEIGHTS, OUTPUT[frame+1], OUTPUT[frame])
		
		if frame == 1:
			# predictions/errors
			obj_pred = return_buffer(OUTPUT[frame][OBJ_PRED_IND])
			cat_pred = return_buffer(OUTPUT[frame][CAT_PRED_IND])
			
			obj_err += return_buffer(OUTPUT[frame][OBJ_OUT_IND])[0]
			cat_err += return_buffer(OUTPUT[frame][CAT_OUT_IND])[0]
			
			obj_class += (objs == obj_pred.argmax(1).squeeze()).sum()
			cat_class += (cats == cat_pred.argmax(1).squeeze()).sum()
			
			# reverse
			WEIGHT_DERIVS_OBJ = reverse_network_btt(OBJ_OUT_IND, LAYERS, WEIGHTS, OUTPUT, WEIGHT_DERIVS_OBJ, frame, abort_layer=abort_obj)
			WEIGHT_DERIVS_CAT = reverse_network_btt(CAT_OUT_IND, LAYERS, WEIGHTS, OUTPUT, WEIGHT_DERIVS_CAT, frame, abort_layer=abort_cat)
			
			WEIGHT_DERIVS_RMS_OBJ = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_RMS_OBJ, EPS / BATCH_SZ, batch, batch_LAG)
			WEIGHT_DERIVS_RMS_CAT = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_CAT, EPS / BATCH_SZ, batch, batch_LAG)
			
			if train_filters_on != 0:
				break
		
		if frame >= 2: # test phase
			cur_err = return_buffer(OUTPUT[frame+1][OUT_IND])[0]
			
			err += cur_err
			err_t_series[frame] += cur_err
			
			WEIGHT_DERIVS = reverse_network_btt(OUT_IND, LAYERS, WEIGHTS, OUTPUT, WEIGHT_DERIVS, frame+1)#, reset_derivs=False)
			WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS / (N_FUTURE*BATCH_SZ), batch, batch_LAG)

	##############
	# print/save/test
	if batch % SAVE_FREQ == 0 and batch != 0:
		if train_filters_on == 0:
			for frame in range(3):
				output_buffer[frame] = return_buffer(OUTPUT[frame+1][PRED_IND])
		target_buffer = copy.deepcopy(frame_target)
		input_buffer = copy.deepcopy(movie_inputs)
		
		###########
		# test movies
		for t_batch in range(N_BATCHES_TEST_MOVIE):
			###############
			# forward movie
			for frame in range(3):
				objs, cats, cat_target, obj_target, movie_inputs, frame_target = load_movie_seqs(t_batch, frame, CAT_DIFF_IND, OBJ_DIFF_IND, DIFF_IND, PX_INDS, WEIGHTS, testing=True)
				OUTPUT[frame+1] = forward_network(LAYERS, WEIGHTS, OUTPUT[frame+1], OUTPUT[frame])
				
				if frame == 1:
					# predictions/errors
					obj_pred = return_buffer(OUTPUT[frame][OBJ_PRED_IND])
					cat_pred = return_buffer(OUTPUT[frame][CAT_PRED_IND])
					
					obj_test_err += return_buffer(OUTPUT[frame][OBJ_OUT_IND])[0]
					cat_test_err += return_buffer(OUTPUT[frame][CAT_OUT_IND])[0]
					
					obj_test_class += (objs == obj_pred.argmax(1).squeeze()).sum()
					cat_test_class += (cats == cat_pred.argmax(1).squeeze()).sum()
					
					if train_filters_on != 0:
						break
					
				if frame >= 2: # test phase
					cur_err = return_buffer(OUTPUT[frame+1][OUT_IND])[0]
					
					err_test += cur_err
					err_t_series_test[frame] += cur_err
		
		# collect test predictions
		if train_filters_on == 0:
			for frame in range(3):
				output_buffer_test[frame] = return_buffer(OUTPUT[frame+1][PRED_IND])
			
		######## log/save
		err_log.append(err / (BATCH_SZ*SAVE_FREQ*N_FUTURE)); err = 0
		
		cat_err_log.append(cat_err / (BATCH_SZ*SAVE_FREQ)); cat_err = 0;
		cat_class_log.append(np.single(cat_class) / (BATCH_SZ*SAVE_FREQ)); cat_class = 0;
		
		obj_err_log.append(obj_err / (BATCH_SZ*SAVE_FREQ)); obj_err = 0;
		obj_class_log.append(np.single(obj_class) / (BATCH_SZ*SAVE_FREQ)); obj_class = 0;
		
		err_t_series_log.append(err_t_series / (BATCH_SZ*SAVE_FREQ/EPOCH_LEN))
		err_t_series = np.zeros(EPOCH_LEN, dtype='single')
		
		#
		err_test_log.append(err_test / (BATCH_SZ*N_BATCHES_TEST_MOVIE*N_FUTURE)); err_test = 0
		
		cat_test_err_log.append(cat_test_err / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); cat_test_err = 0;
		cat_test_class_log.append(np.single(cat_test_class) / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); cat_test_class = 0;
		
		obj_test_err_log.append(obj_test_err / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); obj_test_err = 0;
		obj_test_class_log.append(np.single(obj_test_class) / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); obj_test_class = 0;
		
		err_t_series_test_log.append(err_t_series_test / (BATCH_SZ*N_BATCHES_TEST_MOVIE/EPOCH_LEN))
		err_t_series_test = np.zeros(EPOCH_LEN, dtype='single')
		
		print 'batch: ', (np.single(batch * BATCH_SZ) / (N_MOVIES*MOVIE_FILE_SZ)), batch, 'time: ', time.time() - t_start, 'GPU:', GPU_IND, save_name
		print
		if train_filters_on == 0:
			print 'err: ', err_log[-1], 'err_test: ', err_test_log[-1]
			print
		print 'obj_class: ', obj_class_log[-1], 'obj_err: ', obj_err_log[-1]
		print 'obj_class: ', obj_test_class_log[-1], 'obj_err: ', obj_test_err_log[-1]
		print
		print 'cat_class: ', cat_class_log[-1], 'cat_err: ', cat_err_log[-1]
		print 'cat_class: ', cat_test_class_log[-1], 'cat_err: ', cat_test_err_log[-1]
		print '------------'
		WEIGHTS_F1 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])
		
		savemat('/home/darren/' + save_name + '.mat', {'N_MOVIES': N_MOVIES, 'err_t_series_log': err_t_series_log,
			'output_buffer': output_buffer, 'target_buffer': target_buffer, 'input_buffer': input_buffer,
			'output_buffer_test': output_buffer_test, 'target_buffer_test': frame_target, 'input_buffer_test': movie_inputs,
			'err_test_log': err_test_log, 'err_log': err_log,
			'cat_err_log': cat_err_log, 'cat_class_log': cat_class_log, 'obj_err_log': obj_err_log, 'obj_class_log': obj_class_log,\
			'cat_test_err_log': cat_test_err_log, 'cat_test_class_log': cat_test_class_log, 'obj_test_err_log': obj_test_err_log, 'obj_test_class_log': obj_test_class_log,\
			'F1': WEIGHTS_F1, 'EPS': EPS, 'N_FUTURE': N_FUTURE})
			
		t_start = time.time()
		
	batch += 1
