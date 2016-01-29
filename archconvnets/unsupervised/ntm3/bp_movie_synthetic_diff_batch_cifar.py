# todo: save script; cifar opt
import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr

DIFF = False
DIFF = True

N_FUTURE = 1 # how far into the future to predict
N_CTT = 3 # number of frames to use with Conv. Through Time model (CTT)
TIME_STEPS_PER_MOVIE = 50 # measure of how slow movement is (higher = slower)
N_MOVIES = 4681
BATCH_SZ = 100

EPS = 1e-1 / BATCH_SZ

from architectures.model_architecture_movie32_ctt import *
save_name = 'synthetic_ctt_%f_n_future_%i_%it_batched_cifar' % (-EPS, N_FUTURE, TIME_STEPS_PER_MOVIE)
	
if DIFF:
	save_name += '_diff'

save_name += '_constback'

free_all_buffers()

################ init save vars

frame = 0; frame_local = 0; err = 0; corr = 0; movie_ind = 0; 
cifar_err = 0; cifar_class = 0
cat_err = 0; cat_class = 0; obj_err = 0; obj_class = 0

EPOCH_LEN = 11 # length of movie
SAVE_FREQ = 100*BATCH_SZ #50 #250 # instantaneous checkpoint
WRITE_FREQ = 100*BATCH_SZ # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ
STOP_POINT = np.inf #SAVE_FREQ*15

target_buffer = np.zeros((SAVE_FREQ, N_FRAMES_PRED*N_IN),dtype='single')
output_buffer = np.zeros((SAVE_FREQ, N_FRAMES_PRED*N_IN),dtype='single')
err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

STACK_SUM_IND = find_layer(LAYERS, 'STACK_SUM5') # network frame prediction
TARGET_IND = find_layer(LAYERS, 'ERR') # frame target

CIFAR_PRED_IND = find_layer(LAYERS, 'CIFAR') # cifar prediction vector
SYN_OBJ_PRED_IND = find_layer(LAYERS, 'SYN_OBJ')
SYN_CAT_PRED_IND = find_layer(LAYERS, 'SYN_CAT')

CIFAR_DIFF_IND = find_layer(LAYERS, 'CIFAR_DIFF') # cifar target
SYN_CAT_DIFF_IND = find_layer(LAYERS, 'SYN_CAT_DIFF')
SYN_OBJ_DIFF_IND = find_layer(LAYERS, 'SYN_OBJ_DIFF')

CIFAR_OUT_IND = find_layer(LAYERS, 'CIFAR_ERR') # cifar error
SYN_OBJ_OUT_IND = find_layer(LAYERS, 'SYN_OBJ_ERR')
SYN_CAT_OUT_IND = find_layer(LAYERS, 'SYN_CAT_ERR')

F1_IND = 0

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT = None; WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None
OUTPUT_CIFAR = None; WEIGHT_DERIVS_CIFAR = None; WEIGHT_DERIVS_RMS_CIFAR = None
OUTPUT_OBJ = None; WEIGHT_DERIVS_OBJ = None; WEIGHT_DERIVS_RMS_OBJ = None
OUTPUT_CAT = None; WEIGHT_DERIVS_CAT = None; WEIGHT_DERIVS_RMS_CAT = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

WEIGHTS_F1_INIT = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])

################## load cifar
N_IMGS_CIFAR = 50000

z2 = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(1))
for batch in range(2,6):
	y = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
	z2['data'] = np.concatenate((z2['data'], y['data']), axis=1)
	z2['labels'] = np.concatenate((z2['labels'], y['labels']))
		
x = np.single(z2['data'])/(z2['data'].max()) - .5
cifar_imgs = np.ascontiguousarray(np.single(x.reshape((3, 32, 32, N_IMGS_CIFAR))).transpose((3,0,1,2))[:,np.newaxis])

labels_cifar = np.asarray(z2['labels'])
l = np.zeros((N_IMGS_CIFAR, 10),dtype='uint8')
l[np.arange(N_IMGS_CIFAR),np.asarray(z2['labels']).astype(int)] = 1
Y_cifar = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

set_buffer(Y_cifar[0], WEIGHTS[CIFAR_DIFF_IND][1]) # cifar target

def train_cifar(cifar_err, cifar_class, WEIGHTS, OUTPUT_CIFAR, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR):
	cifar_frame = frame % N_IMGS_CIFAR
	
	temp = np.tile(cifar_imgs[cifar_frame], (N_CTT,1,1,1)).reshape((1,N_CTT*3, IM_SZ, IM_SZ))
	set_buffer(temp, WEIGHTS[F1_IND][1])  # cifar input
	set_buffer(Y_cifar[cifar_frame], WEIGHTS[CIFAR_DIFF_IND][1]) # cifar target
	
	OUTPUT_CIFAR = forward_network(LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV)
	
	cifar_pred = return_buffer(OUTPUT[CIFAR_PRED_IND])
	cifar_err += return_buffer(OUTPUT[CIFAR_OUT_IND])
	cifar_class += np.argmax(Y_cifar[cifar_frame]) == np.argmax(cifar_pred)
	
	WEIGHT_DERIVS_CIFAR = reverse_network(CIFAR_OUT_IND, LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CIFAR, reset_derivs=False)
	
	if frame % BATCH_SZ == 0 and frame != 0:
		#WEIGHT_DERIVS_RMS_CIFAR = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR, -EPS, frame, FRAME_LAG)
		WEIGHTS = update_weights(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, -EPS)
		zero_buffer_list(WEIGHT_DERIVS_CIFAR)
	
	return cifar_err, cifar_class, WEIGHTS, OUTPUT_CIFAR, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR

def train_synth(obj_err, cat_err, obj_class, cat_class, OUTPUT, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_OBJ, WEIGHT_DERIVS_RMS_CAT):
	obj_pred = return_buffer(OUTPUT[SYN_OBJ_PRED_IND])
	cat_pred = return_buffer(OUTPUT[SYN_CAT_PRED_IND])
	
	obj_err += return_buffer(OUTPUT[SYN_OBJ_OUT_IND])
	cat_err += return_buffer(OUTPUT[SYN_CAT_OUT_IND])
	
	obj_class += obj == np.argmax(obj_pred)
	cat_class += cat == np.argmax(cat_pred)
	
	WEIGHT_DERIVS_OBJ = reverse_network(SYN_OBJ_OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_OBJ, abort_layer='F3_MAX',reset_derivs=False)
	WEIGHT_DERIVS_CAT = reverse_network(SYN_CAT_OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CAT, abort_layer='F3_MAX',reset_derivs=False)
	
	if frame % BATCH_SZ == 0 and frame != 0:
		WEIGHT_DERIVS_RMS_OBJ = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_RMS_OBJ, -EPS, frame, FRAME_LAG)
		WEIGHT_DERIVS_RMS_CAT = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_CAT, -EPS, frame, FRAME_LAG)
		zero_buffer_list(WEIGHT_DERIVS_OBJ)
		zero_buffer_list(WEIGHT_DERIVS_CAT)
	
	return obj_err, cat_err, obj_class, cat_class, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_OBJ, WEIGHT_DERIVS_RMS_CAT

#####################
while True:
	
	#### new movie
	movie_name = '/home/darren/rotating_objs32_constback_' + str(TIME_STEPS_PER_MOVIE) + 't/imgs' + str(np.random.randint(N_MOVIES))  + '.mat'
	z = loadmat(movie_name)
	
	cat = z['cat'][0][0]; obj = z['obj'][0][0]
	inputs = np.ascontiguousarray(np.single(z['imgs'] - .5))
	
	cat_target = np.zeros((8,1), dtype='single'); cat_target[cat] = 1
	obj_target = np.zeros((32,1), dtype='single'); obj_target[obj] = 1
	
	set_buffer(cat_target, WEIGHTS[SYN_CAT_DIFF_IND][1])
	set_buffer(obj_target, WEIGHTS[SYN_OBJ_DIFF_IND][1])
	
	frame_local = np.random.randint(EPOCH_LEN - N_FUTURE - N_FRAMES_PRED + 1 - N_CTT) + N_CTT

	###### forward
	if DIFF:
		frame_target = (inputs[frame_local] - inputs[frame_local+N_FUTURE:frame_local+N_FUTURE+N_FRAMES_PRED]).ravel()[:,np.newaxis]
	else:
		frame_target = inputs[frame_local+N_FUTURE:frame_local+N_FUTURE+N_FRAMES_PRED].ravel()[:,np.newaxis]
	frame_target = np.ascontiguousarray(frame_target)
	
	set_buffer(inputs[frame_local-N_CTT:frame_local].reshape((1,N_CTT*3, IM_SZ, IM_SZ)), WEIGHTS[F1_IND][1])  # inputs
	set_buffer(frame_target, WEIGHTS[TARGET_IND][1]) # frame target
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	time_series_prediction = return_buffer(OUTPUT[STACK_SUM_IND]).ravel()
	err += return_buffer(OUTPUT[TARGET_IND])
	corr += pearsonr(frame_target.ravel(), time_series_prediction.ravel())[0]

	output_buffer[frame % SAVE_FREQ] = copy.deepcopy(time_series_prediction)
	target_buffer[frame % SAVE_FREQ] = copy.deepcopy(frame_target.ravel())
	
	####### cat gradient step
	cifar_err, cifar_class, WEIGHTS, OUTPUT_CIFAR, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR = train_cifar(cifar_err, cifar_class, WEIGHTS, OUTPUT_CIFAR, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR)
	obj_err, cat_err, obj_class, cat_class, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_OBJ, WEIGHT_DERIVS_RMS_CAT = train_synth(obj_err, cat_err, obj_class, cat_class, OUTPUT, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_OBJ, WEIGHT_DERIVS_RMS_CAT)
	
	# print/save
	if frame % SAVE_FREQ == 0 and frame != 0:
		corr_log.append(corr / SAVE_FREQ); corr = 0
		err_log.append(err / SAVE_FREQ); err = 0
		cifar_err_log.append(cifar_err / SAVE_FREQ); cifar_err = 0;
		cifar_class_log.append([np.single(cifar_class) / SAVE_FREQ]); cifar_class = 0;
		cat_err_log.append(cat_err / SAVE_FREQ); cat_err = 0;
		cat_class_log.append([np.single(cat_class) / SAVE_FREQ]); cat_class = 0;
		obj_err_log.append(obj_err / SAVE_FREQ); obj_err = 0;
		obj_class_log.append([np.single(obj_class) / SAVE_FREQ]); obj_class = 0;
		
		
		print_state(LAYERS, WEIGHTS, WEIGHT_DERIVS, OUTPUT, EPS, err_log, frame, corr_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names)
		save_conv_state(LAYERS, WEIGHTS, WEIGHTS_F1_INIT, save_name, output_buffer, target_buffer, EPS, err_log, corr_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, EPOCH_LEN)
		
		t_start = time.time()
		
	frame += 1

free_list_list(MEM_DERIVS)
free_partials(PARTIALS_PREV)
free_list(OUTPUT)
free_list(WEIGHT_DERIVS)
free_list(OUTPUT_PREV)
