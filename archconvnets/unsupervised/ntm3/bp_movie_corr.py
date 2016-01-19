# todo: save script; cifar opt
import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr

no_mem = True
no_mem = False
T_AHEAD = 2

if no_mem:
	from architectures.model_architecture_movie_no_mem import init_model
	INPUT_SCALE = 1e-5
	EPS = 5e-4
	save_name = 'ntm_movie_corr_no_mem_%f_T_AHEAD_%i' % (-EPS, T_AHEAD)
else:
	from architectures.model_architecture_movie import init_model
	INPUT_SCALE = 1e-5
	EPS = 5e-4
	save_name = 'ntm_movie_corr_%f_T_AHEAD_%i' % (-EPS, T_AHEAD)

	
free_all_buffers()


################ init save vars

frame = 0; frame_local = 0; err = 0

EPOCH_LEN = 2000
SAVE_FREQ = 50 # instantaneous checkpoint
WRITE_FREQ = 50 # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ
STOP_POINT = np.inf #SAVE_FREQ*15

output_buffer = np.zeros(SAVE_FREQ,dtype='single')
err_log = []; corr_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

STACK_SUM_IND = find_layer(LAYERS, 'STACK_SUM')
TARGET_IND = find_layer(LAYERS, 'ERR')
OUT_IND = find_layer(LAYERS, 'ERR')
F1_IND = 0

OUTPUT = None; WEIGHT_DERIVS = None
MEM_DERIVS = [None]*len(MEM_INDS); WEIGHT_DERIVS_RMS = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

WEIGHTS_F1_INIT = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])

############# load images
z = loadmat('/home/darren/test_clip.mat')
imgs = z['imgs'] * INPUT_SCALE
imgs -= imgs.mean(0)[np.newaxis] # subtract mean

inputs = np.single(np.ascontiguousarray(imgs[:,np.newaxis])) # (n_imgs, 1, 3,32,32)
targets = imgs.reshape((imgs.shape[0],3*32*32,1))

#####################
while True:
	# show the movie again
	if (frame % EPOCH_LEN) == 0:
		free_list(OUTPUT_PREV)
		free_partials(PARTIALS_PREV)
		
		OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
		PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
		
		frame_local = 0
	
	###### forward
	set_buffer(inputs[frame_local], WEIGHTS[F1_IND][1])  # inputs
	set_buffer(targets[frame_local+T_AHEAD], WEIGHTS[TARGET_IND][1]) # target
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	current_err = return_buffer(OUTPUT[-1])
	err += current_err;  output_buffer[frame % SAVE_FREQ]

	
	###### reverse
	WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
		
	# update partials_prev
	MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
	PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
	
	OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	# take step
	if frame < STOP_POINT and frame > SAVE_FREQ:
		WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS, frame, FRAME_LAG)
		
		
	# print
	if frame % SAVE_FREQ == 0 and frame != 0:
		corr_log.append(pearsonr(targets[frame_local+T_AHEAD], return_buffer(OUTPUT[STACK_SUM_IND]))[0])
		err_log.append(err / SAVE_FREQ); err = 0
		
		print_state(LAYERS, WEIGHTS, WEIGHT_DERIVS, OUTPUT, EPS, err_log, frame, corr_log, t_start, save_name, print_names)
		
		#######
		WEIGHTS_F1 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])
		WEIGHTS_F2 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F2')][0])
		WEIGHTS_F3 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F3')][0])
		savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'err_log': err_log, 'corr_log': corr_log, 'EPS': EPS, \
				'F1_init': WEIGHTS_F1_INIT, 'F1': WEIGHTS_F1, 'F2': WEIGHTS_F2, 'F3': WEIGHTS_F3})
		
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