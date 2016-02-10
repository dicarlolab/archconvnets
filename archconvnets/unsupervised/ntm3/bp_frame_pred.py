import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from architectures.ctt_frame_pred import *
from img_sets import *

EPS = 1e-1

DIFF = True
train_filters_on = 0

abort_cifar = abort_cat = abort_obj = abort_imgnet = 'F3_MAX'
MOVIE_BREAK_LAYER = 'OBJ_SUM_ERR'
if train_filters_on == 0:
	save_name = 'frame_pred_N_FUTURE_%i' % N_FUTURE
	MOVIE_BREAK_LAYER = 'SUM_ERR'
elif train_filters_on == 1:
	save_name = 'cifar'
	abort_cifar = None
elif train_filters_on == 2:
	save_name = 'cat'
	abort_cat = None
elif train_filters_on == 3:
	save_name = 'obj'
	abort_obj = None
elif train_filters_on == 4:
	save_name = 'imgnet'
	abort_imgnet = None
else:
	save_name = 'rand'
 
save_name += '_EPS_%f_N_CTT_%i_N_MOVIES_%i' % (EPS, N_CTT, N_MOVIES)

if DIFF:
	save_name += '_diff'

free_all_buffers()

################ init save vars
batch = 0; err = 0; err_test = 0; corr = 0; corr_test = 0; movie_ind = 0; 
cifar_err = 0; cifar_class = 0; cifar_test_err = 0; cifar_test_class = 0;
imgnet_err = 0; imgnet_class = 0; imgnet_test_err = 0; imgnet_test_class = 0
cat_err = 0; cat_class = 0; cat_test_err = 0; cat_test_class = 0; 
obj_err = 0; obj_class = 0; obj_test_err = 0; obj_test_class = 0

SAVE_FREQ = 10000/(BATCH_SZ) # instantaneous checkpoint
batch_LAG = 100 #SAVE_FREQ

train_prediction = test_prediction = train_target = test_target = 0
test_inputs = train_inputs = 0

err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
err_test_log = []; corr_test_log = []; cifar_test_err_log = []; cifar_test_class_log = []
imgnet_err_log = []; imgnet_class_log = []
imgnet_test_err_log = []; imgnet_test_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []
cat_test_err_log = []; cat_test_class_log = []; obj_test_err_log = []; obj_test_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()

PRED_IND = find_layer(LAYERS, 'STACK_SUM2')
IMGNET_PRED_IND = find_layer(LAYERS, 'IMGNET')
CIFAR_PRED_IND = find_layer(LAYERS, 'CIFAR')
OBJ_PRED_IND = find_layer(LAYERS, 'OBJ')
CAT_PRED_IND = find_layer(LAYERS, 'CAT')

DIFF_IND = find_layer(LAYERS, 'ERR')
IMGNET_DIFF_IND = find_layer(LAYERS, 'IMGNET_ERR')
CIFAR_DIFF_IND = find_layer(LAYERS, 'CIFAR_ERR')
CAT_DIFF_IND = find_layer(LAYERS, 'CAT_ERR')
OBJ_DIFF_IND = find_layer(LAYERS, 'OBJ_ERR')
IMGNET_DIFF_IND = find_layer(LAYERS, 'IMGNET_ERR')

OUT_IND = find_layer(LAYERS, 'SUM_ERR')
IMGNET_OUT_IND = find_layer(LAYERS, 'IMGNET_SUM_ERR')
CIFAR_OUT_IND = find_layer(LAYERS, 'CIFAR_SUM_ERR')
OBJ_OUT_IND = find_layer(LAYERS, 'OBJ_SUM_ERR')
CAT_OUT_IND = find_layer(LAYERS, 'CAT_SUM_ERR')

MOVIE_BREAK_LAYER_IND = find_layer(LAYERS, MOVIE_BREAK_LAYER)

F1_IND = 0

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT = None; WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None
OUTPUT_CIFAR = None; WEIGHT_DERIVS_CIFAR = None; WEIGHT_DERIVS_RMS_CIFAR = None
OUTPUT_IMGNET = None; WEIGHT_DERIVS_IMGNET = None; WEIGHT_DERIVS_RMS_IMGNET = None
OUTPUT_OBJ = None; WEIGHT_DERIVS_OBJ = None; WEIGHT_DERIVS_RMS_OBJ = None
OUTPUT_CAT = None; WEIGHT_DERIVS_CAT = None; WEIGHT_DERIVS_RMS_CAT = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

#####################
while True:
	#######
	# load imgs
	cifar_target, cifar_inputs = load_cifar(batch, N_CTT)
	imgnet_target, imgnet_inputs = load_imgnet(batch, N_CTT)
	objs, cats, cat_target, obj_target, movie_inputs, frame_target = load_movies(N_CTT, DIFF=DIFF)
	
	#######
	# set targets
	set_buffer(cat_target, WEIGHTS[CAT_DIFF_IND][1])
	set_buffer(obj_target, WEIGHTS[OBJ_DIFF_IND][1])
	set_buffer(cifar_target, WEIGHTS[CIFAR_DIFF_IND][1])
	set_buffer(imgnet_target, WEIGHTS[IMGNET_DIFF_IND][1])
	set_buffer(frame_target, WEIGHTS[DIFF_IND][1])
	
	##############
	# forward cifar
	set_buffer(cifar_inputs, WEIGHTS[F1_IND][1])
	OUTPUT_CIFAR = forward_network(LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV, break_layer=CIFAR_OUT_IND)
	
	# predictions/errors
	cifar_pred = return_buffer(OUTPUT_CIFAR[CIFAR_PRED_IND])
	cifar_err += return_buffer(OUTPUT_CIFAR[CIFAR_OUT_IND])[0]
	cifar_class += (cifar_target.argmax(1).squeeze() == cifar_pred.argmax(1).squeeze()).sum()
	
	# reverse
	WEIGHT_DERIVS_CIFAR = reverse_network(CIFAR_OUT_IND, LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CIFAR, abort_layer=abort_cifar)
	WEIGHT_DERIVS_RMS_CIFAR = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR, EPS / BATCH_SZ, batch, batch_LAG)
	
	##############
	# forward imgnet
	set_buffer(imgnet_inputs, WEIGHTS[F1_IND][1])
	OUTPUT_IMGNET = forward_network(LAYERS, WEIGHTS, OUTPUT_IMGNET, OUTPUT_PREV, break_layer=IMGNET_OUT_IND)
	
	# predictions/errors
	imgnet_pred = return_buffer(OUTPUT_IMGNET[IMGNET_PRED_IND])
	imgnet_err += return_buffer(OUTPUT_IMGNET[IMGNET_OUT_IND])[0]
	imgnet_class += (imgnet_target.argmax(1).squeeze() == imgnet_pred.argmax(1).squeeze()).sum()
	
	# reverse
	WEIGHT_DERIVS_IMGNET = reverse_network(IMGNET_OUT_IND, LAYERS, WEIGHTS, OUTPUT_IMGNET, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_IMGNET, abort_layer=abort_imgnet)
	WEIGHT_DERIVS_RMS_IMGNET = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_IMGNET, WEIGHT_DERIVS_RMS_IMGNET, EPS / BATCH_SZ, batch, batch_LAG)
	
	###############
	# forward movie
	set_buffer(movie_inputs, WEIGHTS[F1_IND][1])
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
		
		##########
		# test imgnet/cifar
		for t_batch in range(N_BATCHES_TEST):
			cifar_target, cifar_inputs = load_cifar(t_batch, N_CTT, testing=True)
			imgnet_target, imgnet_inputs = load_imgnet(t_batch, N_CTT, testing=True)
			
			set_buffer(cifar_target, WEIGHTS[CIFAR_DIFF_IND][1])
			set_buffer(imgnet_target, WEIGHTS[IMGNET_DIFF_IND][1])
			
			##############
			# forward cifar
			set_buffer(cifar_inputs, WEIGHTS[F1_IND][1])
			OUTPUT_CIFAR = forward_network(LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV, break_layer=CIFAR_OUT_IND)
			
			# predictions/errors
			cifar_pred = return_buffer(OUTPUT_CIFAR[CIFAR_PRED_IND])
			cifar_test_err += return_buffer(OUTPUT_CIFAR[CIFAR_OUT_IND])[0]
			cifar_test_class += (cifar_target.argmax(1).squeeze() == cifar_pred.argmax(1).squeeze()).sum()
			
			##############
			# forward imgnet
			set_buffer(imgnet_inputs, WEIGHTS[F1_IND][1])
			OUTPUT_IMGNET = forward_network(LAYERS, WEIGHTS, OUTPUT_IMGNET, OUTPUT_PREV, break_layer=IMGNET_OUT_IND)
			
			# predictions/errors
			imgnet_pred = return_buffer(OUTPUT_IMGNET[IMGNET_PRED_IND])
			imgnet_test_err += return_buffer(OUTPUT_IMGNET[IMGNET_OUT_IND])[0]
			imgnet_test_class += (imgnet_target.argmax(1).squeeze() == imgnet_pred.argmax(1).squeeze()).sum()
			
		###########
		# test movies
		for t_batch in range(N_BATCHES_TEST_MOVIE):
			objs, cats, cat_target, obj_target, movie_inputs, frame_target = load_movies(N_CTT, DIFF=DIFF, testing=t_batch)
			
			set_buffer(cat_target, WEIGHTS[CAT_DIFF_IND][1])
			set_buffer(obj_target, WEIGHTS[OBJ_DIFF_IND][1])
			set_buffer(frame_target, WEIGHTS[DIFF_IND][1])
			
			###############
			# forward movie
			set_buffer(movie_inputs, WEIGHTS[F1_IND][1])
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
		
		cifar_err_log.append(cifar_err / (BATCH_SZ*SAVE_FREQ)); cifar_err = 0;
		cifar_class_log.append(np.single(cifar_class) / (BATCH_SZ*SAVE_FREQ)); cifar_class = 0;
		
		imgnet_err_log.append(imgnet_err / (BATCH_SZ*SAVE_FREQ)); imgnet_err = 0;
		imgnet_class_log.append(np.single(imgnet_class) / (BATCH_SZ*SAVE_FREQ)); imgnet_class = 0;
		
		cat_err_log.append(cat_err / (BATCH_SZ*SAVE_FREQ)); cat_err = 0;
		cat_class_log.append(np.single(cat_class) / (BATCH_SZ*SAVE_FREQ)); cat_class = 0;
		
		obj_err_log.append(obj_err / (BATCH_SZ*SAVE_FREQ)); obj_err = 0;
		obj_class_log.append(np.single(obj_class) / (BATCH_SZ*SAVE_FREQ)); obj_class = 0;
		
		#
		corr_test_log.append(corr_test / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); corr_test = 0
		err_test_log.append(err_test / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); err_test = 0
		
		cifar_test_err_log.append(cifar_test_err / (BATCH_SZ*N_BATCHES_TEST)); cifar_test_err = 0;
		cifar_test_class_log.append(np.single(cifar_test_class) / (BATCH_SZ*N_BATCHES_TEST)); cifar_test_class = 0;
		
		imgnet_test_err_log.append(imgnet_test_err / (BATCH_SZ*N_BATCHES_TEST)); imgnet_test_err = 0;
		imgnet_test_class_log.append(np.single(imgnet_test_class) / (BATCH_SZ*N_BATCHES_TEST)); imgnet_test_class = 0;
		
		cat_test_err_log.append(cat_test_err / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); cat_test_err = 0;
		cat_test_class_log.append(np.single(cat_test_class) / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); cat_test_class = 0;
		
		obj_test_err_log.append(obj_test_err / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); obj_test_err = 0;
		obj_test_class_log.append(np.single(obj_test_class) / (BATCH_SZ*N_BATCHES_TEST_MOVIE)); obj_test_class = 0;
		
		print 'batch: ', (np.single(batch * BATCH_SZ) / 50000), 'time: ', time.time() - t_start, 'GPU:', GPU_IND, save_name
		print
		if train_filters_on == 0:
			print 'err: ', err_log[-1], 'err_test: ', err_test_log[-1]
			print
		print 'imgnet_class: ', imgnet_class_log[-1], 'imgnet_err: ', imgnet_err_log[-1]
		print 'imgnet_class: ', imgnet_test_class_log[-1], 'imgnet_err: ', imgnet_test_err_log[-1]
		print 
		print 'cifar_class: ', cifar_class_log[-1], 'cifar_err: ', cifar_err_log[-1]
		print 'cifar_class: ', cifar_test_class_log[-1], 'cifar_err: ', cifar_test_err_log[-1]
		print
		print 'obj_class: ', obj_class_log[-1], 'obj_err: ', obj_err_log[-1]
		print 'obj_class: ', obj_test_class_log[-1], 'obj_err: ', obj_test_err_log[-1]
		print
		print 'cat_class: ', cat_class_log[-1], 'cat_err: ', cat_err_log[-1]
		print 'cat_class: ', cat_test_class_log[-1], 'cat_err: ', cat_test_err_log[-1]
		print '------------'
		WEIGHTS_F1 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])
		WEIGHTS_F2 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F2')][0])
		WEIGHTS_F3 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F3')][0])
		
		savemat('/home/darren/' + save_name + '.mat', {'output_buffer': [], 'target_buffer': [], 'N_MOVIES': N_MOVIES, \
			'err_test_log': err_test_log, 'corr_test_log': corr_test_log, 'cifar_test_err_log': cifar_test_err_log, \
			'cifar_test_class_log': cifar_test_class_log, \
			'test_prediction': test_prediction, 'test_target': test_target, 'test_inputs': test_inputs,
			'train_prediction': train_prediction, 'train_target': train_target, 'train_inputs': train_inputs,
			'imgnet_test_err_log': imgnet_test_err_log, 'imgnet_test_class_log': imgnet_test_class_log, \
			'err_log': err_log, 'corr_log': corr_log, 'cifar_err_log': cifar_err_log, 'cifar_class_log': cifar_class_log, \
			'imgnet_err_log': imgnet_err_log, 'imgnet_class_log': imgnet_class_log, 'movie_inputs': movie_inputs, \
			'cat_err_log': cat_err_log, 'cat_class_log': cat_class_log, 'obj_err_log': obj_err_log, 'obj_class_log': obj_class_log,\
			'cat_test_err_log': cat_test_err_log, 'cat_test_class_log': cat_test_class_log, 'obj_test_err_log': obj_test_err_log, 'obj_test_class_log': obj_test_class_log,\
			'F1': WEIGHTS_F1, 'F2': WEIGHTS_F2, 'F3': WEIGHTS_F3, 'EPOCH_LEN': EPOCH_LEN, 'EPS': EPS})
			
		t_start = time.time()
		
	batch += 1
