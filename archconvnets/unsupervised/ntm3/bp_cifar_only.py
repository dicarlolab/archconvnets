import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from architectures.cifar_only import *

EPS = 1e-1

train_filters_on = 0

save_name = 'cifar_%f' % (EPS)

free_all_buffers()

################ init save vars

frame = 0; frame_local = 0; err = 0; corr = 0; movie_ind = 0; 
cifar_err = 0; cifar_class = 0
cat_err = 0; cat_class = 0; obj_err = 0; obj_class = 0

SAVE_FREQ = 10000/(BATCH_SZ) # instantaneous checkpoint
WRITE_FREQ = 10000/(BATCH_SZ) # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ
STOP_POINT = np.inf

err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

CIFAR_PRED_IND = find_layer(LAYERS, 'CIFAR')
CIFAR_DIFF_IND = find_layer(LAYERS, 'CIFAR_ERR')
CIFAR_OUT_IND = find_layer(LAYERS, 'CIFAR_SUM_ERR')

F1_IND = 0

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT_CIFAR = None; WEIGHT_DERIVS_CIFAR = None; WEIGHT_DERIVS_RMS_CIFAR = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

WEIGHTS_F1_INIT = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])

################## load cifar
N_IMGS_CIFAR = 50000

z2 = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(1))
imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
for batch in range(2,6):
	y = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
	z2['data'] = np.concatenate((z2['data'], y['data']), axis=1)
	z2['labels'] = np.concatenate((z2['labels'], y['labels']))
	
x = z2['data'] - imgs_mean
x = x.reshape((3, 32, 32, 50000))
cifar_imgs = np.ascontiguousarray(x.transpose((3,0,1,2))[:,np.newaxis])

labels_cifar = np.asarray(z2['labels'])
l = np.zeros((N_IMGS_CIFAR, 10),dtype='uint8')
l[np.arange(N_IMGS_CIFAR),np.asarray(z2['labels']).astype(int)] = 1
Y_cifar = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

set_buffer(Y_cifar[:BATCH_SZ], WEIGHTS[CIFAR_DIFF_IND][1]) # cifar target

#####################
while True:
	cifar_batch = frame % (N_IMGS_CIFAR / BATCH_SZ)
	
	# load targets
	cifar_target = Y_cifar[cifar_batch*BATCH_SZ:(cifar_batch+1)*BATCH_SZ]
	
	set_buffer(cifar_target, WEIGHTS[CIFAR_DIFF_IND][1])
	
	# forward cifar
	cifar_inputs = np.tile(cifar_imgs[cifar_batch*BATCH_SZ:(cifar_batch+1)*BATCH_SZ], (1,N_CTT,1,1,1)).reshape((BATCH_SZ,N_CTT*3, IM_SZ, IM_SZ))
	set_buffer(cifar_inputs, WEIGHTS[F1_IND][1])  # cifar input
	OUTPUT_CIFAR = forward_network(LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV)
	
	# predictions/errors
	cifar_pred = return_buffer(OUTPUT_CIFAR[CIFAR_PRED_IND])
	cifar_err += return_buffer(OUTPUT_CIFAR[CIFAR_OUT_IND])
	cifar_class += (cifar_target.argmax(1).squeeze() == cifar_pred.argmax(1).squeeze()).sum()
	
	# reverse
	WEIGHT_DERIVS_CIFAR = reverse_network(CIFAR_OUT_IND, LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CIFAR)
	WEIGHT_DERIVS_RMS_CIFAR = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR, EPS / BATCH_SZ, frame, FRAME_LAG)
	#WEIGHTS = update_weights(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, -EPS / BATCH_SZ)
	
	
	# print/save
	if frame % SAVE_FREQ == 0 and frame != 0:
		corr_log.append([corr / (BATCH_SZ*SAVE_FREQ)]); corr = 0
		err_log.append([err / (BATCH_SZ*SAVE_FREQ)]); err = 0
		cifar_err_log.append(cifar_err / (BATCH_SZ*SAVE_FREQ)); cifar_err = 0;
		cifar_class_log.append([np.single(cifar_class) / (BATCH_SZ*SAVE_FREQ)]); cifar_class = 0;
		cat_err_log.append([cat_err / (BATCH_SZ*SAVE_FREQ)]); cat_err = 0;
		cat_class_log.append([np.single(cat_class) / (BATCH_SZ*SAVE_FREQ)]); cat_class = 0;
		obj_err_log.append([obj_err / (BATCH_SZ*SAVE_FREQ)]); obj_err = 0;
		obj_class_log.append([np.single(obj_class) / (BATCH_SZ*SAVE_FREQ)]); obj_class = 0;
		
		print_state(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, OUTPUT_CIFAR, EPS, err_log, (np.single(frame * BATCH_SZ) / 50000), corr_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names)
		save_conv_state(LAYERS, WEIGHTS, WEIGHTS_F1_INIT, save_name, [], [], EPS, err_log, corr_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, 1, 1)
		
		t_start = time.time()
		
	frame += 1
