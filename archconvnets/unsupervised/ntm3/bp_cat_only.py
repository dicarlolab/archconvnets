import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from architectures.ctt_categorization_only import *

EPS = 1e-1

train_filters_on = 4

abort_cifar = abort_cat = abort_obj = abort_imgnet = 'F3_MAX'
if train_filters_on == 0:
	save_name = 'cifar'
	abort_cifar = None
elif train_filters_on == 1:
	save_name = 'cat'
	abort_cat = None
elif train_filters_on == 2:
	save_name = 'obj'
	abort_obj = None
elif train_filters_on == 3:
	save_name = 'imgnet'
	abort_imgnet = None
else:
	save_name = 'rand'
 
save_name += '_%i_N_CTT_%f' % (N_CTT, EPS)

free_all_buffers()

################ init save vars

frame = 0; frame_local = 0; err = 0; corr = 0; movie_ind = 0; 
cifar_err = 0; cifar_class = 0; imgnet_err = 0; imgnet_class = 0
cat_err = 0; cat_class = 0; obj_err = 0; obj_class = 0

EPOCH_LEN = 11 # length of movie
SAVE_FREQ = 10000/(BATCH_SZ) # instantaneous checkpoint
WRITE_FREQ = 10000/(BATCH_SZ) # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ
STOP_POINT = np.inf

err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
imgnet_err_log = []; imgnet_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

IMGNET_PRED_IND = find_layer(LAYERS, 'IMGNET')
CIFAR_PRED_IND = find_layer(LAYERS, 'CIFAR')
OBJ_PRED_IND = find_layer(LAYERS, 'OBJ')
CAT_PRED_IND = find_layer(LAYERS, 'CAT')

IMGNET_DIFF_IND = find_layer(LAYERS, 'IMGNET_ERR')
CIFAR_DIFF_IND = find_layer(LAYERS, 'CIFAR_ERR')
CAT_DIFF_IND = find_layer(LAYERS, 'CAT_ERR')
OBJ_DIFF_IND = find_layer(LAYERS, 'OBJ_ERR')
IMGNET_DIFF_IND = find_layer(LAYERS, 'IMGNET_ERR')

IMGNET_OUT_IND = find_layer(LAYERS, 'IMGNET_SUM_ERR')
CIFAR_OUT_IND = find_layer(LAYERS, 'CIFAR_SUM_ERR')
OBJ_OUT_IND = find_layer(LAYERS, 'OBJ_SUM_ERR')
CAT_OUT_IND = find_layer(LAYERS, 'CAT_SUM_ERR')

F1_IND = 0

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT = None; WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None
OUTPUT_CIFAR = None; WEIGHT_DERIVS_CIFAR = None; WEIGHT_DERIVS_RMS_CIFAR = None
OUTPUT_IMGNET = None; WEIGHT_DERIVS_IMGNET = None; WEIGHT_DERIVS_RMS_IMGNET = None
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

z2['data'] = np.single(z2['data'])
z2['data'] /= z2['data'].max()
mean_img = z2['data'].mean(1)[:,np.newaxis]
x = z2['data'] - mean_img
cifar_imgs = np.ascontiguousarray(np.single(x.reshape((3, 32, 32, N_IMGS_CIFAR))).transpose((3,0,1,2))[:,np.newaxis])

labels_cifar = np.asarray(z2['labels'])
l = np.zeros((N_IMGS_CIFAR, 10),dtype='uint8')
l[np.arange(N_IMGS_CIFAR),np.asarray(z2['labels']).astype(int)] = 1
Y_cifar = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

set_buffer(Y_cifar[:BATCH_SZ], WEIGHTS[CIFAR_DIFF_IND][1]) # cifar target
mean_img = mean_img.reshape((1,1,3,32,32))

########## movie init
movie_inputs = np.zeros((BATCH_SZ, N_CTT*3, 32, 32), dtype='single')

cats = np.zeros(BATCH_SZ)
objs = np.zeros(BATCH_SZ)

###################### imgnet
N_IMGNET_FILES = 199
IMGNET_FILE_SZ = 10000
set_buffer(np.single(np.random.random((BATCH_SZ, 999,1))), WEIGHTS[IMGNET_DIFF_IND][1])

#####################
while True:
	cifar_batch = frame % (N_IMGS_CIFAR / BATCH_SZ)
	imgnet_batch = frame % (IMGNET_FILE_SZ / BATCH_SZ)
	
	imgnet_file = ((frame*BATCH_SZ)/IMGNET_FILE_SZ) % N_IMGNET_FILES
	
	# load imgnet
	if imgnet_batch == 0:
		z = loadmat('/home/darren/imgnet32/data_batch_' + str(imgnet_file+1))
		imgnet_imgs = np.single(z['data'])
		imgnet_imgs /= imgnet_imgs.max()
		imgnet_imgs = imgnet_imgs.reshape((IMGNET_FILE_SZ, 1, 3, IM_SZ, IM_SZ))
		imgnet_imgs -= mean_img
		
		labels_imgnet = z['labels'].squeeze()
		l = np.zeros((IMGNET_FILE_SZ, 999),dtype='uint8')
		l[np.arange(IMGNET_FILE_SZ), labels_imgnet.astype(int)] = 1
		Y_imgnet = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories
	
	
	# load movies
	cat_target = np.zeros((BATCH_SZ, 8, 1), dtype='single')
	obj_target = np.zeros((BATCH_SZ, 32, 1), dtype='single')
	
	for img in range(BATCH_SZ):
		movie_frame = np.random.randint(EPOCH_LEN - N_CTT) + N_CTT # movies
		z = loadmat('/home/darren/rotating_objs32_constback_50t/imgs' + str(np.random.randint(N_MOVIES))  + '.mat')
		
		cats[img] = z['cat'][0][0]
		objs[img] = z['obj'][0][0]
		
		cat_target[img, cats[img]] = 1
		obj_target[img, objs[img]] = 1
		
		movie_inputs[img] = (z['imgs'][movie_frame-N_CTT:movie_frame] - mean_img).reshape((1,N_CTT*3, IM_SZ, IM_SZ))
	
	movie_inputs = np.ascontiguousarray(movie_inputs)
	
	#######
	# load targets
	cifar_target = Y_cifar[cifar_batch*BATCH_SZ:(cifar_batch+1)*BATCH_SZ]
	imgnet_target = Y_imgnet[imgnet_batch*BATCH_SZ:(imgnet_batch+1)*BATCH_SZ]
	
	set_buffer(cat_target, WEIGHTS[CAT_DIFF_IND][1])
	set_buffer(obj_target, WEIGHTS[OBJ_DIFF_IND][1])
	set_buffer(cifar_target, WEIGHTS[CIFAR_DIFF_IND][1])
	set_buffer(imgnet_target, WEIGHTS[IMGNET_DIFF_IND][1])
	
	###############
	# forward movie
	set_buffer(movie_inputs, WEIGHTS[F1_IND][1])
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
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
	
	WEIGHT_DERIVS_RMS_OBJ = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_RMS_OBJ, EPS / BATCH_SZ, frame, FRAME_LAG)
	WEIGHT_DERIVS_RMS_CAT = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_CAT, EPS / BATCH_SZ, frame, FRAME_LAG)
	
	##############
	# forward cifar
	cifar_inputs = np.tile(cifar_imgs[cifar_batch*BATCH_SZ:(cifar_batch+1)*BATCH_SZ], (1,N_CTT,1,1,1)).reshape((BATCH_SZ,N_CTT*3, IM_SZ, IM_SZ))
	set_buffer(cifar_inputs, WEIGHTS[F1_IND][1])
	OUTPUT_CIFAR = forward_network(LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV)
	
	# predictions/errors
	cifar_pred = return_buffer(OUTPUT_CIFAR[CIFAR_PRED_IND])
	cifar_err += return_buffer(OUTPUT_CIFAR[CIFAR_OUT_IND])[0]
	cifar_class += (cifar_target.argmax(1).squeeze() == cifar_pred.argmax(1).squeeze()).sum()
	
	# reverse
	WEIGHT_DERIVS_CIFAR = reverse_network(CIFAR_OUT_IND, LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CIFAR, abort_layer=abort_cifar)
	WEIGHT_DERIVS_RMS_CIFAR = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR, EPS / BATCH_SZ, frame, FRAME_LAG)
	
	##############
	# forward imgnet
	imgnet_inputs = np.tile(imgnet_imgs[imgnet_batch*BATCH_SZ:(imgnet_batch+1)*BATCH_SZ], (1,N_CTT,1,1,1)).reshape((BATCH_SZ,N_CTT*3, IM_SZ, IM_SZ))
	set_buffer(imgnet_inputs, WEIGHTS[F1_IND][1])
	OUTPUT_IMGNET = forward_network(LAYERS, WEIGHTS, OUTPUT_IMGNET, OUTPUT_PREV)
	
	# predictions/errors
	imgnet_pred = return_buffer(OUTPUT_IMGNET[IMGNET_PRED_IND])
	imgnet_err += return_buffer(OUTPUT_IMGNET[IMGNET_OUT_IND])[0]
	imgnet_class += (imgnet_target.argmax(1).squeeze() == imgnet_pred.argmax(1).squeeze()).sum()
	
	# reverse
	WEIGHT_DERIVS_IMGNET = reverse_network(IMGNET_OUT_IND, LAYERS, WEIGHTS, OUTPUT_IMGNET, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_IMGNET, abort_layer=abort_imgnet)
	WEIGHT_DERIVS_RMS_IMGNET = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_IMGNET, WEIGHT_DERIVS_RMS_IMGNET, EPS / BATCH_SZ, frame, FRAME_LAG)
	
	# print/save
	if frame % SAVE_FREQ == 0 and frame != 0:
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
		
		print_state(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, OUTPUT_CIFAR, EPS, err_log, (np.single(frame * BATCH_SZ) / 50000), corr_log, cifar_err_log, cifar_class_log, imgnet_err_log, imgnet_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names)
		save_conv_state(LAYERS, WEIGHTS, WEIGHTS_F1_INIT, save_name, [], [], EPS, err_log, corr_log, cifar_err_log, cifar_class_log, imgnet_err_log, imgnet_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, EPOCH_LEN, N_MOVIES)
		
		t_start = time.time()
		
	frame += 1
