import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from architectures.ctt_frame_pred_seq import *
from img_sets.movie_seqs import *

EPS = 1e-1

train_filters_on = 0

abort_cat = abort_obj = 'F3_MAX'
MOVIE_BREAK_LAYER = 'OBJ_SUM_ERR'
if train_filters_on == 0:
	save_name = 'frame_pred_N_FUTURE_%i' % N_FUTURE
	MOVIE_BREAK_LAYER = 'SUM_ERR'
elif train_filters_on == 1:
	save_name = 'cat'
	abort_cat = None
elif train_filters_on == 2:
	save_name = 'obj'
	abort_obj = None
else:
	save_name = 'rand'
 
save_name += '_EPS_%f_N_CTT_%i_N_MOVIES_%i' % (EPS, N_CTT, N_MOVIES)

if DIFF:
	save_name += '_diff_2000_32F'

free_all_buffers()

################ init save vars
batch = 0; err = 0; err_test = 0; corr = 0; corr_test = 0; movie_ind = 0; 
cat_err = 0; cat_class = 0; cat_test_err = 0; cat_test_class = 0; 
obj_err = 0; obj_class = 0; obj_test_err = 0; obj_test_class = 0

SAVE_FREQ = 10000/(BATCH_SZ) # instantaneous checkpoint
batch_LAG = 100

train_prediction = test_prediction = train_target = test_target = 0
test_inputs = train_inputs = 0

err_log = []; corr_log = []
err_test_log = []; corr_test_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []
cat_test_err_log = []; cat_test_class_log = []; obj_test_err_log = []; obj_test_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()

PRED_IND = find_layer(LAYERS, 'STACK_SUM2')
OBJ_PRED_IND = find_layer(LAYERS, 'OBJ')
CAT_PRED_IND = find_layer(LAYERS, 'CAT')

DIFF_IND = find_layer(LAYERS, 'ERR')
CAT_DIFF_IND = find_layer(LAYERS, 'CAT_ERR')
OBJ_DIFF_IND = find_layer(LAYERS, 'OBJ_ERR')

OUT_IND = find_layer(LAYERS, 'SUM_ERR')
OBJ_OUT_IND = find_layer(LAYERS, 'OBJ_SUM_ERR')
CAT_OUT_IND = find_layer(LAYERS, 'CAT_SUM_ERR')

MOVIE_BREAK_LAYER_IND = find_layer(LAYERS, MOVIE_BREAK_LAYER)

F1_IND = 0

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT = None; WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None
OUTPUT_OBJ = None; WEIGHT_DERIVS_OBJ = None; WEIGHT_DERIVS_RMS_OBJ = None
OUTPUT_CAT = None; WEIGHT_DERIVS_CAT = None; WEIGHT_DERIVS_RMS_CAT = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

#####################
while True:
	###############
	# forward movie
	objs, cats, cat_target, obj_target, movie_inputs, frame_target = load_movie_seqs(batch, N_CTT, CAT_DIFF_IND, OBJ_DIFF_IND, DIFF_IND, F1_IND, WEIGHTS, DIFF=DIFF)
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, break_layer=MOVIE_BREAK_LAYER_IND)
	
	# predictions/errors
	obj_pred = return_buffer(OUTPUT[OBJ_PRED_IND])
	cat_pred = return_buffer(OUTPUT[CAT_PRED_IND])
	
	obj_err += return_buffer(OUTPUT[OBJ_OUT_IND])[0]
	cat_err += return_buffer(OUTPUT[CAT_OUT_IND])[0]
	
	obj_class += (objs == obj_pred.argmax(1).squeeze()).sum()
	cat_class += (cats == cat_pred.argmax(1).squeeze()).sum()
	
	# reverse
	WEIGHT_DERIVS_OBJ = reverse_network(OBJ_OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_OBJ, abort_layer=abort_obj)
	WEIGHT_DERIVS_CAT = reverse_network(CAT_OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CAT, abort_layer=abort_cat)
	
	WEIGHT_DERIVS_RMS_OBJ = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_RMS_OBJ, EPS / BATCH_SZ, batch, batch_LAG)
	WEIGHT_DERIVS_RMS_CAT = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_CAT, EPS / BATCH_SZ, batch, batch_LAG)
	
	if train_filters_on == 0:
		err += return_buffer(OUTPUT[OUT_IND])[0]
		WEIGHT_DERIVS = reverse_network(OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
		WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS / BATCH_SZ, batch, batch_LAG)
	
	##############
	# print/save/test
	if batch % SAVE_FREQ == 0 and batch != 0:
		if train_filters_on == 0:
			train_prediction = return_buffer(OUTPUT[PRED_IND])
			train_target = copy.deepcopy(frame_target)
			train_inputs = copy.deepcopy(movie_inputs)
		
		###########
		# test movies
		for t_batch in range(N_BATCHES_TEST_MOVIE):
			
			###############
			# forward movie
			objs, cats, cat_target, obj_target, movie_inputs, frame_target = load_movie_seqs(t_batch, N_CTT, CAT_DIFF_IND, OBJ_DIFF_IND, DIFF_IND, F1_IND, WEIGHTS, DIFF=DIFF, testing=t_batch)
			OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, break_layer=MOVIE_BREAK_LAYER_IND)
			
			# predictions/errors
			obj_pred = return_buffer(OUTPUT[OBJ_PRED_IND])
			cat_pred = return_buffer(OUTPUT[CAT_PRED_IND])
			
			obj_test_err += return_buffer(OUTPUT[OBJ_OUT_IND])[0]
			cat_test_err += return_buffer(OUTPUT[CAT_OUT_IND])[0]
			
			obj_test_class += (objs == obj_pred.argmax(1).squeeze()).sum()
			cat_test_class += (cats == cat_pred.argmax(1).squeeze()).sum()
			
			if train_filters_on == 0:
				test_prediction = return_buffer(OUTPUT[PRED_IND])
				test_target = frame_target
				test_inputs = movie_inputs
				
				err_test += return_buffer(OUTPUT[OUT_IND])[0]
		
		######## log/save
		corr_log.append(corr / (BATCH_SZ*SAVE_FREQ)); corr = 0
		err_log.append(err / (BATCH_SZ*SAVE_FREQ)); err = 0
		
		cat_err_log.append(cat_err / (BATCH_SZ*SAVE_FREQ)); cat_err = 0;
		cat_class_log.append(np.single(cat_class) / (BATCH_SZ*SAVE_FREQ)); cat_class = 0;
		
		obj_err_log.append(obj_err / (BATCH_SZ*SAVE_FREQ)); obj_err = 0;
		obj_class_log.append(np.single(obj_class) / (BATCH_SZ*SAVE_FREQ)); obj_class = 0;
		
		#
		corr_test_log.append(corr_test / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); corr_test = 0
		err_test_log.append(err_test / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); err_test = 0
		
		cat_test_err_log.append(cat_test_err / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); cat_test_err = 0;
		cat_test_class_log.append(np.single(cat_test_class) / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); cat_test_class = 0;
		
		obj_test_err_log.append(obj_test_err / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); obj_test_err = 0;
		obj_test_class_log.append(np.single(obj_test_class) / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); obj_test_class = 0;
		
		print 'batch: ', (np.single(batch * BATCH_SZ) / (N_MOVIES*MOVIE_FILE_SZ)), 'time: ', time.time() - t_start, 'GPU:', GPU_IND, save_name
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
		
		savemat('/home/darren/' + save_name + '.mat', {'output_buffer': [], 'target_buffer': [], 'N_MOVIES': N_MOVIES, \
			'err_test_log': err_test_log, 'corr_test_log': corr_test_log, 
			'test_prediction': test_prediction, 'test_target': test_target, 'test_inputs': test_inputs,
			'train_prediction': train_prediction, 'train_target': train_target, 'train_inputs': train_inputs,
			'err_log': err_log, 'corr_log': corr_log, 'movie_inputs': movie_inputs, \
			'cat_err_log': cat_err_log, 'cat_class_log': cat_class_log, 'obj_err_log': obj_err_log, 'obj_class_log': obj_class_log,\
			'cat_test_err_log': cat_test_err_log, 'cat_test_class_log': cat_test_class_log, 'obj_test_err_log': obj_test_err_log, 'obj_test_class_log': obj_test_class_log,\
			'F1': WEIGHTS_F1, 'EPOCH_LEN': EPOCH_LEN, 'EPS': EPS})
			
		t_start = time.time()
		
	batch += 1
