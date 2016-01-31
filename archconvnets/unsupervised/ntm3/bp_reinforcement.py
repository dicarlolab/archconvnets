import numpy as np
import time
import scipy.optimize
from ntm_core import *
from scipy.io import loadmat, savemat
from scipy.stats import zscore, pearsonr
from architectures.reinforcement import *
from worlds.panda_world import *

EPS = 1e-1
EPS_GREED_FINAL_TIME = 4*5000000

save_name = 'reinforcement_%f_EPS_%i_FIN_TIME' % (EPS, EPS_GREED_FINAL_TIME)

free_all_buffers()

################ init save vars
N_MOVIES = 6372
BATCH_SZ = 32
MEM_SZ = 1000000
EPS_GREED_FINAL = .1
GAMMA = 0.99
BATCH_SZ = 32
NETWORK_UPDATE = 10000
EPOCH_LEN = 11 # length of movie
SAVE_FREQ = 1000 #50 #250 # instantaneous checkpoint
FRAME_LAG = 100 #SAVE_FREQ
GAME_TRAIN_FRAC = 1#3 # train on the game X times more than categorization (speed reasons)

frame = 0; err = 0;  r_total = 0
cifar_err = 0; cifar_class = 0
cat_err = 0; cat_class = 0; obj_err = 0; obj_class = 0
network_updates = 0

r_log = []; err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()
WEIGHTS_PREV = copy_weights(WEIGHTS)

CIFAR_PRED_IND = find_layer(LAYERS, 'CIFAR')
SYN_OBJ_PRED_IND = find_layer(LAYERS, 'SYN_OBJ')
SYN_CAT_PRED_IND = find_layer(LAYERS, 'SYN_CAT')

CIFAR_DIFF_IND = find_layer(LAYERS, 'CIFAR_ERR')
SYN_CAT_DIFF_IND = find_layer(LAYERS, 'SYN_CAT_ERR')
SYN_OBJ_DIFF_IND = find_layer(LAYERS, 'SYN_OBJ_ERR')

CIFAR_OUT_IND = find_layer(LAYERS, 'CIFAR_ERR')
SYN_OBJ_OUT_IND = find_layer(LAYERS, 'SYN_OBJ_ERR')
SYN_CAT_OUT_IND = find_layer(LAYERS, 'SYN_CAT_ERR')

GAME_PRED_IND = [None]*N_C; GAME_DIFF_IND = [None]*N_C; GAME_OUT_IND = [None]*N_C;
for action in range(N_C):
	GAME_PRED_IND[action] = find_layer(LAYERS, 'GAME_PRED_'+str(action))
	GAME_DIFF_IND[action] = find_layer(LAYERS, 'SUM_'+str(action))
	GAME_OUT_IND[action] = find_layer(LAYERS, 'SUM_ERR_'+str(action))

F1_IND = 0

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT_PREV_NETWORK = None
OUTPUT = None; WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None
OUTPUT_CIFAR = None; WEIGHT_DERIVS_CIFAR = None; WEIGHT_DERIVS_RMS_CIFAR = None
OUTPUT_MOVIE = None; WEIGHT_DERIVS_OBJ = None; WEIGHT_DERIVS_RMS_OBJ = None
WEIGHT_DERIVS_CAT = None; WEIGHT_DERIVS_RMS_CAT = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

WEIGHTS_F1_INIT = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])

# set null targets for game action output predictions
set_buffer(np.zeros((8,1), dtype='single'), WEIGHTS[SYN_CAT_DIFF_IND][1])
set_buffer(np.zeros((8,1), dtype='single'), WEIGHTS_PREV[SYN_CAT_DIFF_IND][1])
set_buffer(np.zeros((32,1), dtype='single'), WEIGHTS[SYN_OBJ_DIFF_IND][1])
set_buffer(np.zeros((32,1), dtype='single'), WEIGHTS_PREV[SYN_OBJ_DIFF_IND][1])
set_buffer(np.zeros((10,1), dtype='single'), WEIGHTS[CIFAR_DIFF_IND][1])
set_buffer(np.zeros((10,1), dtype='single'), WEIGHTS_PREV[CIFAR_DIFF_IND][1])
for action in range(N_C):
	set_buffer(np.zeros((1,1),dtype='single'), WEIGHTS[GAME_DIFF_IND[action]][1])
	set_buffer(np.zeros((1,1),dtype='single'), WEIGHTS_PREV[GAME_DIFF_IND[action]][1])

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

################## buffers for game replay
panda_coords_input = np.zeros((MEM_SZ, N_PANDAS, 2), dtype='single')
kid_coords_input = np.zeros((MEM_SZ, N_KIDS, 2), dtype='single')

panda_directions_input = np.zeros((MEM_SZ, N_PANDAS), dtype='single')
kid_directions_input = np.zeros((MEM_SZ, N_KIDS), dtype='single')

x_input = np.zeros(MEM_SZ, dtype='single')
y_input = np.zeros(MEM_SZ, dtype='single')
direction_input = np.zeros(MEM_SZ, dtype='single')

action_input = np.zeros(MEM_SZ, dtype='int')

##
panda_coords_output = np.zeros((MEM_SZ, N_PANDAS, 2), dtype='single')
kid_coords_output = np.zeros((MEM_SZ, N_KIDS, 2), dtype='single')

panda_directions_output = np.zeros((MEM_SZ, N_PANDAS), dtype='single')
kid_directions_output = np.zeros((MEM_SZ, N_KIDS), dtype='single')

x_output = np.zeros(MEM_SZ, dtype='single')
y_output = np.zeros(MEM_SZ, dtype='single')
direction_output = np.zeros(MEM_SZ, dtype='single')

##
r_output = np.zeros(MEM_SZ, dtype='single')
y_outputs = np.zeros(MEM_SZ, dtype='single')
y_network_ver = -np.ones(MEM_SZ, dtype='single')

##
panda_coords_recent = np.zeros((SAVE_FREQ, N_PANDAS, 2), dtype='single')
kid_coords_recent = np.zeros((SAVE_FREQ, N_KIDS, 2), dtype='single')

panda_directions_recent = np.zeros((SAVE_FREQ, N_PANDAS), dtype='single')
kid_directions_recent = np.zeros((SAVE_FREQ, N_KIDS), dtype='single')

x_recent = np.zeros(SAVE_FREQ, dtype='single')
y_recent = np.zeros(SAVE_FREQ, dtype='single')
direction_recent = np.zeros(SAVE_FREQ, dtype='single')

action_recent = np.zeros(SAVE_FREQ, dtype='int')
r_recent = np.zeros(SAVE_FREQ, dtype='single')

pred = np.zeros(N_C, dtype='single')
pred_prev = np.zeros(N_C, dtype='single')

##
x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions = init_pos_vars()


#####################
while True:
	mem_loc = frame % MEM_SZ
	movie_frame = np.random.randint(EPOCH_LEN - N_CTT) + N_CTT # movies
	cifar_frame = frame % N_IMGS_CIFAR
	
	if frame >= MEM_SZ and frame % GAME_TRAIN_FRAC == 0:
		###############
		# classification training
		
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
		set_buffer(Y_cifar[cifar_frame], WEIGHTS[CIFAR_DIFF_IND][1])
		
		# forward cifar
		temp = np.tile(cifar_imgs[cifar_frame], (N_CTT,1,1,1)).reshape((1,N_CTT*3, IM_SZ, IM_SZ))
		set_buffer(temp, WEIGHTS[F1_IND][1])  # cifar input
		OUTPUT_CIFAR = forward_network(LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV)
		
		# forward movie
		set_buffer(inputs[movie_frame-N_CTT:movie_frame].reshape((1,N_CTT*3, IM_SZ, IM_SZ)), WEIGHTS[F1_IND][1])  # inputs
		OUTPUT_MOVIE = forward_network(LAYERS, WEIGHTS, OUTPUT_MOVIE, OUTPUT_PREV)
		
		# save classification predictions/errors
		cifar_pred = return_buffer(OUTPUT_CIFAR[CIFAR_PRED_IND])
		obj_pred = return_buffer(OUTPUT_MOVIE[SYN_OBJ_PRED_IND])
		cat_pred = return_buffer(OUTPUT_MOVIE[SYN_CAT_PRED_IND])
		
		obj_err += return_buffer(OUTPUT_MOVIE[SYN_OBJ_OUT_IND])
		cat_err += return_buffer(OUTPUT_MOVIE[SYN_CAT_OUT_IND])
		cifar_err += return_buffer(OUTPUT_CIFAR[CIFAR_OUT_IND])
		
		obj_class += obj == np.argmax(obj_pred)
		cat_class += cat == np.argmax(cat_pred)
		cifar_class += np.argmax(Y_cifar[cifar_frame]) == np.argmax(cifar_pred)
		
	###########
	# game
	
	# copy current game input state
	panda_coords_input[mem_loc] = copy.deepcopy(panda_coords)
	kid_coords_input[mem_loc] = copy.deepcopy(kid_coords)

	panda_directions_input[mem_loc] = copy.deepcopy(panda_directions)
	kid_directions_input[mem_loc] = copy.deepcopy(kid_directions)

	x_input[mem_loc] = x
	y_input[mem_loc] = y
	direction_input[mem_loc] = direction
	
	# choose game action
	network_outputs_computed = False
	CHANCE_RAND = np.max((1 - ((1-EPS_GREED_FINAL)/EPS_GREED_FINAL_TIME)*(frame - MEM_SZ), EPS_GREED_FINAL))
	if np.random.rand() <= CHANCE_RAND:
		action = np.random.randint(3)
		if action >= 3: # increase likelihood of movement
			action = 0
	else:
		img = np.single(render(x,y, direction, panda, kid, kid_coords, panda_coords, kid_directions, panda_directions)) / 255
		network_outputs_computed = True
		
		# forward pass
		set_buffer(img - .5, WEIGHTS[F1_IND][1])  # inputs
		
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		
		for action in range(N_C):
			pred[action] = return_buffer(OUTPUT[GAME_PRED_IND[action]])
		action = np.argmax(pred)
		
	# perform action
	r = 0
	if action == 0:
		r, x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions = move(x,y, direction, kid_coords, panda_coords, kid_directions, panda_directions)
	elif action == 1:
		direction += ROT_RATE
	elif action == 2:
		direction -= ROT_RATE
	
	r_total += r
	
	# copy current state
	panda_coords_output[mem_loc] = copy.deepcopy(panda_coords)
	kid_coords_output[mem_loc] = copy.deepcopy(kid_coords)

	panda_directions_output[mem_loc] = copy.deepcopy(panda_directions)
	kid_directions_output[mem_loc] = copy.deepcopy(kid_directions)

	x_output[mem_loc] = x
	y_output[mem_loc] = y
	direction_output[mem_loc] = direction
	
	r_output[mem_loc] = r
	action_input[mem_loc] = action
	
	if network_outputs_computed == True: # otherwise, the input wasn't actually fed through
		y_outputs[mem_loc] = r + GAMMA * np.max(pred)
		y_network_ver[mem_loc] = network_updates % NETWORK_UPDATE
	else:
		y_network_ver[mem_loc] = -1
	
	# debug/for visualizations
	save_loc = frame % SAVE_FREQ
	
	panda_coords_recent[save_loc] = copy.deepcopy(panda_coords)
	kid_coords_recent[save_loc] = copy.deepcopy(kid_coords)

	panda_directions_recent[save_loc] = copy.deepcopy(panda_directions)
	kid_directions_recent[save_loc] = copy.deepcopy(kid_directions)

	x_recent[save_loc] = x
	y_recent[save_loc] = y
	direction_recent[save_loc] = direction

	action_recent[save_loc] = action
	r_recent[save_loc] = r

	if frame == MEM_SZ:
		print 'beginning gradient computations'
	
	######################################
	# update gradient?
	if frame >= MEM_SZ:
		trans = np.random.randint(MEM_SZ)

		# forward pass prev network
		# (only compute if we have not already computed the output for this version of the network)
		if y_network_ver[trans] != (network_updates % NETWORK_UPDATE):
			img_prev = np.single(render(x_output[trans],y_output[trans], direction_output[trans], panda, kid, kid_coords_output[trans], \
				panda_coords_output[trans], kid_directions_output[trans], panda_directions_output[trans])) / 255
			
			set_buffer(img_prev - .5, WEIGHTS_PREV[F1_IND][1])  # inputs
			
			OUTPUT_PREV_NETWORK = forward_network(LAYERS, WEIGHTS_PREV, OUTPUT_PREV_NETWORK, OUTPUT_PREV)
			
			for action in range(N_C):
				pred_prev[action] = return_buffer(OUTPUT_PREV_NETWORK[GAME_PRED_IND[action]])
			
			# save output:
			y_outputs[trans] = r_output[trans] + GAMMA * np.max(pred_prev)
			y_network_ver[trans] = network_updates % NETWORK_UPDATE
			
		# forward pass current network
		img_cur = np.single(render(x_input[trans],y_input[trans], direction_input[trans], panda, kid, kid_coords_input[trans], \
			panda_coords_input[trans], kid_directions_input[trans], panda_directions_input[trans])) / 255
		
		set_buffer(img_cur - .5, WEIGHTS[F1_IND][1])  # inputs
		set_buffer(y_outputs[trans].reshape((1,1)), WEIGHTS[GAME_DIFF_IND[action_input[trans]]][1]) # target
		
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		
		err += return_buffer(OUTPUT[GAME_OUT_IND[action_input[trans]]])
		
		########### backprop
		WEIGHT_DERIVS = reverse_network(GAME_OUT_IND[action_input[trans]], LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS, reset_derivs=False)
		if frame % GAME_TRAIN_FRAC == 0:
			WEIGHT_DERIVS_OBJ = reverse_network(SYN_OBJ_OUT_IND, LAYERS, WEIGHTS, OUTPUT_MOVIE, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_OBJ, abort_layer='F3_MAX', reset_derivs=False)
			WEIGHT_DERIVS_CAT = reverse_network(SYN_CAT_OUT_IND, LAYERS, WEIGHTS, OUTPUT_MOVIE, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CAT, abort_layer='F3_MAX', reset_derivs=False)
			WEIGHT_DERIVS_CIFAR = reverse_network(CIFAR_OUT_IND, LAYERS, WEIGHTS, OUTPUT_CIFAR, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS_CIFAR, abort_layer='F3_MAX', reset_derivs=False)
		
		#### update filter weights
		if(frame - MEM_SZ) % BATCH_SZ == 0 and frame != MEM_SZ:
			WEIGHT_DERIVS_RMS = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, -EPS / BATCH_SZ, frame, FRAME_LAG)
			WEIGHT_DERIVS_RMS_OBJ = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_OBJ, WEIGHT_DERIVS_RMS_OBJ, EPS / BATCH_SZ, frame, FRAME_LAG)
			WEIGHT_DERIVS_RMS_CAT = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CAT, WEIGHT_DERIVS_RMS_CAT, EPS / BATCH_SZ, frame, FRAME_LAG)
			WEIGHT_DERIVS_RMS_CIFAR = update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, WEIGHT_DERIVS_RMS_CIFAR, EPS / BATCH_SZ, frame, FRAME_LAG)
			
			zero_buffer_list(WEIGHT_DERIVS)
			zero_buffer_list(WEIGHT_DERIVS_OBJ)
			zero_buffer_list(WEIGHT_DERIVS_CAT)
			zero_buffer_list(WEIGHT_DERIVS_CIFAR)
			
			network_updates += 1
			
			if network_updates % NETWORK_UPDATE == 0:
				print 'updating network'
				WEIGHTS_PREV = copy_weights(WEIGHTS, WEIGHTS_PREV)
	
	################
	
	# print/save
	if frame % SAVE_FREQ == 0 and frame != 0 and frame > MEM_SZ:
		r_log.append([r_total]); r_total = 0
		corr_log.append(err / SAVE_FREQ)
		err_log.append(err / SAVE_FREQ); err = 0
		cifar_err_log.append(cifar_err / SAVE_FREQ); cifar_err = 0;
		cifar_class_log.append([np.single(cifar_class) / SAVE_FREQ]); cifar_class = 0;
		cat_err_log.append(cat_err / SAVE_FREQ); cat_err = 0;
		cat_class_log.append([np.single(cat_class) / SAVE_FREQ]); cat_class = 0;
		obj_err_log.append(obj_err / SAVE_FREQ); obj_err = 0;
		obj_class_log.append([np.single(obj_class) / SAVE_FREQ]); obj_class = 0;
		
		print_reinforcement_state(LAYERS, WEIGHTS, WEIGHT_DERIVS_CIFAR, OUTPUT_CIFAR, EPS, CHANCE_RAND, err_log, frame, r_log, cifar_err_log, cifar_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names)
		
		WEIGHTS_F1 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])
		WEIGHTS_F2 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F2')][0])
		WEIGHTS_F3 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F3')][0])
		
		img = render(x,y, direction, panda, kid, kid_coords, panda_coords, kid_directions, panda_directions)
		
		savemat('/home/darren/' + save_name, {'r_total_plot': r_log, 'step': frame, 'img': img, 'err_plot': err_log, \
			'panda_coords_recent': panda_coords_recent, 'kid_coords_recent': kid_coords_recent, \
			'panda_directions_recent': panda_directions_recent, 'kid_directions_recent': kid_directions_recent,
			'x_recent': x_recent, 'y_recent': y_recent, 'direction_recent': direction_recent, 'action_recent': action_recent,
			'r_recent': r_recent, 'frame': frame,'N_MOVIES': N_MOVIES, \
			'err_log': err_log, 'corr_log': corr_log, 'cifar_err_log': cifar_err_log, 'cifar_class_log': cifar_class_log, 'EPS': EPS, \
			'cat_err_log': cat_err_log, 'cat_class_log': cat_class_log, 'obj_err_log': obj_err_log, 'obj_class_log': obj_class_log,\
			'F1_init': WEIGHTS_F1_INIT, 'F1': WEIGHTS_F1, 'F2': WEIGHTS_F2, 'F3': WEIGHTS_F3, 'EPOCH_LEN': EPOCH_LEN})
			
		t_start = time.time()
		
	frame += 1

