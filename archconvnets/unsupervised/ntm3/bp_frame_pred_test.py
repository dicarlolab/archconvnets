import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from architectures.ctt_frame_pred_test import *
import Image

EPS = 1e-1

#DIFF = True
DIFF = False

N_FUTURE = 8 # how far into the future to predict

save_name = 'frame_pred_simp_%f_N_FUTURE_%i_N_CTT_%i' % (EPS, N_FUTURE, N_CTT)

if DIFF:
	save_name += '_diff'

free_all_buffers()

################ init save vars

frame = 0; frame_local = 0; err = 0; corr = 0; movie_ind = 0; 
cifar_err = 0; cifar_class = 0; imgnet_err = 0; imgnet_class = 0
cat_err = 0; cat_class = 0; obj_err = 0; obj_class = 0

EPOCH_LEN = 11 # length of movie
SAVE_FREQ = 10000/(BATCH_SZ) # instantaneous checkpoint
WRITE_FREQ = 10000/(BATCH_SZ) # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ

target_buffer = np.zeros((SAVE_FREQ, N_IN), dtype='single')
output_buffer = np.zeros((SAVE_FREQ, N_IN), dtype='single')

err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
imgnet_err_log = []; imgnet_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

PRED_IND = find_layer(LAYERS, 'PRED')
DIFF_IND = find_layer(LAYERS, 'ERR')
OUT_IND = find_layer(LAYERS, 'SUM_ERR')

F1_IND = 0
PX_IND = find_layer(LAYERS, 'PX_0_lin')

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT = None; WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None
OUTPUT_CIFAR = None; WEIGHT_DERIVS_CIFAR = None; WEIGHT_DERIVS_RMS_CIFAR = None
OUTPUT_IMGNET = None; WEIGHT_DERIVS_IMGNET = None; WEIGHT_DERIVS_RMS_IMGNET = None
OUTPUT_OBJ = None; WEIGHT_DERIVS_OBJ = None; WEIGHT_DERIVS_RMS_OBJ = None
OUTPUT_CAT = None; WEIGHT_DERIVS_CAT = None; WEIGHT_DERIVS_RMS_CAT = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

WEIGHTS_F1_INIT = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])

############## movie init
movie_inputs = np.zeros((BATCH_SZ, N_CTT*3, IM_SZ, IM_SZ), dtype='single')
frame_target = np.zeros((BATCH_SZ, N_TARGET, 1), dtype='single')


#####################
while True:
	# load movies
	for img in range(BATCH_SZ):
		movie_frame = np.random.randint(EPOCH_LEN - N_CTT - N_FUTURE + 1) + N_CTT # movies
		z = loadmat('/home/darren/rotating_objs32_constback_50t/imgs' + str(np.random.randint(N_MOVIES))  + '.mat')
		
		movie_inputs[img] = (z['imgs'][movie_frame-N_CTT:movie_frame] - .5).reshape((1,N_CTT*3, IM_SZ, IM_SZ))
		temp = np.asarray(Image.fromarray(np.uint8(255*z['imgs'][movie_frame-1+N_FUTURE]).reshape((3,32,32)).transpose((1,2,0))).resize((16,16)),dtype='single')/255
		frame_target[img] = (temp.transpose((2,0,1)) - .5).reshape((N_TARGET,1))
	
	movie_inputs = np.ascontiguousarray(movie_inputs)
	frame_target = np.ascontiguousarray(frame_target)
	
	########
	# load targets
	set_buffer(frame_target, WEIGHTS[DIFF_IND][1])
	
	###############
	# forward movie
	set_buffer(movie_inputs, WEIGHTS[F1_IND][1])
	set_buffer(movie_inputs, WEIGHTS[PX_IND][1])
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	# predictions/errors
	err += return_buffer(OUTPUT[OUT_IND])[0]
	
	# reverse
	WEIGHT_DERIVS = reverse_network(OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
	WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS / BATCH_SZ, frame, FRAME_LAG)
	
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
		
		output_buffer = return_buffer(OUTPUT[PRED_IND])
		
		print_state(LAYERS, WEIGHTS, WEIGHT_DERIVS, OUTPUT, EPS, err_log, (np.single(frame * BATCH_SZ) / 50000), corr_log, cifar_err_log, cifar_class_log, imgnet_err_log, imgnet_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names)
		save_conv_state(LAYERS, WEIGHTS, WEIGHTS_F1_INIT, save_name, movie_inputs, output_buffer, frame_target, EPS, err_log, corr_log, cifar_err_log, cifar_class_log, imgnet_err_log, imgnet_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, EPOCH_LEN, N_MOVIES)
		
		t_start = time.time()
		
	frame += 1
