import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from architectures.ctt_categorization_only import *

N_MOVIES = 6372
BATCH_SZ = 50
EPS = 1e-1

train_filters_on = 2

abort_cifar = abort_cat = abort_obj = 'F3_MAX'
if train_filters_on == 0:
	save_name = 'cifar'
	abort_cifar = None
elif train_filters_on == 1:
	save_name = 'cat'
	abort_cat = None
else:
	save_name = 'obj'
	abort_obj = None
 
save_name += '_%f' % (-EPS)

free_all_buffers()

################ init save vars

frame = 0; frame_local = 0; err = 0; corr = 0; movie_ind = 0; 
cifar_err = 0; cifar_class = 0
cat_err = 0; cat_class = 0; obj_err = 0; obj_class = 0

EPOCH_LEN = 11 # length of movie
SAVE_FREQ = 25*BATCH_SZ #50 #250 # instantaneous checkpoint
WRITE_FREQ = 25*BATCH_SZ # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ
STOP_POINT = np.inf #SAVE_FREQ*15

err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

CIFAR_PRED_IND = find_layer(LAYERS, 'CIFAR') # cifar prediction vector
SYN_OBJ_PRED_IND = find_layer(LAYERS, 'SYN_OBJ')
SYN_CAT_PRED_IND = find_layer(LAYERS, 'SYN_CAT')

CIFAR_DIFF_IND = find_layer(LAYERS, 'CIFAR_ERR') # cifar target
SYN_CAT_DIFF_IND = find_layer(LAYERS, 'SYN_CAT_ERR')
SYN_OBJ_DIFF_IND = find_layer(LAYERS, 'SYN_OBJ_ERR')

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

#####################
while True:
	movie_frame = np.random.randint(EPOCH_LEN - N_CTT) + N_CTT # movies
	cifar_frame = frame % N_IMGS_CIFAR
	
	# load movie
	movie_name = '/home/darren/rotating_objs32_constback_50t/imgs' + str(np.random.randint(N_MOVIES))  + '.mat'
	z = loadmat(movie_name)
	
	cat = z['cat'][0][0]; obj = z['obj'][0][0]
	inputs = np.ascontiguousarray(np.single(z['imgs'] - .5))
	
	cat_target = np.zeros((8,1), dtype='single'); cat_target[cat] = 1
	obj_target = np.zeros((32,1), dtype='single'); obj_target[obj] = 1
	
	# load targets
	set_buffer(cat_target, WEIGHTS[SYN_CAT_DIFF_IND][1])
	set_buffer(obj_target, WEIGHTS[SYN_OBJ_DIFF_IND][1])
	set_buffer(Y_cifar[cifar_frame], WEIGHTS[CIFAR_DIFF_IND][1]) # cifar target
	
	# forward cifar
	temp = np.tile(cifar_imgs[cifar_frame], (N_CTT,1,1,1)).reshape((1,N_CTT*3, IM_SZ, IM_SZ))
	set_buffer(temp, WEIGHTS[F1_IND][1])  # cifar input
	OUTPUT_CIFAR = forward_network(LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV)
	
	# forward movie
	set_buffer(inputs[movie_frame-N_CTT:movie_frame].reshape((1,N_CTT*3, IM_SZ, IM_SZ)), WEIGHTS[F1_IND][1])  # inputs
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	# predictions/errors
	cifar_pred = return_buffer(OUTPUT_CIFAR[CIFAR_PRED_IND])
	obj_pred = return_buffer(OUTPUT[SYN_OBJ_PRED_IND])
	cat_pred = return_buffer(OUTPUT[SYN_CAT_PRED_IND])
	
	obj_err += return_buffer(OUTPUT[SYN_OBJ_OUT_IND])
	cat_err += return_buffer(OUTPUT[SYN_CAT_OUT_IND])
	cifar_err += return_buffer(OUTPUT_CIFAR[CIFAR_OUT_IND])
	
	obj_class += obj == np.argmax(obj_pred)
	cat_class += cat == np.argmax(cat_pred)
	cifar_class += np.argmax(Y_cifar[cifar_frame]) == np.argmax(cifar_pred)
	
	# reverse
	WEIGHT_DERIVS_OBJ = reverse_network(SYN_OBJ_OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_OBJ, abort_layer=abort_obj, reset_derivs=False)
	WEIGHT_DERIVS_CAT = reverse_network(SYN_CAT_OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CAT, abort_layer=abort_cat, reset_derivs=False)
	WEIGHT_DERIVS_CIFAR = reverse_network(CIFAR_OUT_IND, LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CIFAR, abort_layer=abort_cifar, reset_derivs=False)
	
	if frame % BATCH_SZ == 0 and frame != 0:
		WEIGHT_DERIVS_RMS_OBJ = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_RMS_OBJ, EPS / BATCH_SZ, frame, FRAME_LAG)
		WEIGHT_DERIVS_RMS_CAT = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_CAT, EPS / BATCH_SZ, frame, FRAME_LAG)
		WEIGHT_DERIVS_RMS_CIFAR = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR, EPS / BATCH_SZ, frame, FRAME_LAG)
		
		zero_buffer_list(WEIGHT_DERIVS_CIFAR)
		zero_buffer_list(WEIGHT_DERIVS_OBJ)
		zero_buffer_list(WEIGHT_DERIVS_CAT)
	
	# print/save
	if frame % SAVE_FREQ == 0 and frame != 0:
		corr_log.append([corr / SAVE_FREQ]); corr = 0
		err_log.append([err / SAVE_FREQ]); err = 0
		cifar_err_log.append(cifar_err / SAVE_FREQ); cifar_err = 0;
		cifar_class_log.append([np.single(cifar_class) / SAVE_FREQ]); cifar_class = 0;
		cat_err_log.append(cat_err / SAVE_FREQ); cat_err = 0;
		cat_class_log.append([np.single(cat_class) / SAVE_FREQ]); cat_class = 0;
		obj_err_log.append(obj_err / SAVE_FREQ); obj_err = 0;
		obj_class_log.append([np.single(obj_class) / SAVE_FREQ]); obj_class = 0;
		
		print_state(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, OUTPUT_CIFAR, EPS, err_log, frame, corr_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names)
		save_conv_state(LAYERS, WEIGHTS, WEIGHTS_F1_INIT, save_name, [], [], EPS, err_log, corr_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, EPOCH_LEN, N_MOVIES)
		
		t_start = time.time()
		
	frame += 1

free_list_list(MEM_DERIVS)
free_partials(PARTIALS_PREV)
free_list(OUTPUT)
free_list(WEIGHT_DERIVS)
free_list(OUTPUT_PREV)
