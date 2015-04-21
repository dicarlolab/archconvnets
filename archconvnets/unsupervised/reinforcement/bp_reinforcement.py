from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
import numexpr as ne
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu
import scipy

RAND_PERIOD = 1000000
MEM_SZ = 50000
EPS_GREED = .85
GAMMA = 0.99
BATCH_SZ = 64
NETWORK_UPDATE = 100000
EPS = 1e-5

SCALE = 4
MAX_LOC = 32/SCALE
N_REDS = 7
N_BLUES = 7

F1_scale = 1e-2
F2_scale = 1e-2
F3_scale = 1e-2
FL_scale = 1e-2

N = 16
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 4 # ...
s1 = 5

N_C = 4 # directions L, R, U, D

file_name = '/home/darren/reinforcement.mat'

max_output_sz3  = 5

GPU_CUR = 0
GPU_PREV = 1

# gpu buffer indices
MAX_OUTPUT1 = 0; DF2_DATA = 1; CONV_OUTPUT1 = 2; DPOOL1 = 3
F1_IND = 4; IMGS_PAD = 5; DF1 = 6; F2_IND = 11
D_UNPOOL2 = 12;F3_IND = 13; MAX_OUTPUT2 = 14; MAX_OUTPUT3 = 15
CONV_OUTPUT1 = 19; CONV_OUTPUT2 = 20; CONV_OUTPUT3 = 21
DF2 = 25; DPOOL2 = 26; DF3_DATA = 27; DPOOL3 = 28; DF3 = 29; FL_PRED = 30;
FL_IND = 31; PRED = 32; DFL = 33

#np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n1, max_output_sz3, max_output_sz3)))

FL_prev = copy.deepcopy(FL)
set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_PREV)

set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_CUR)

F1_init = copy.deepcopy(F1)

dF1 = np.zeros_like(F1)
dF2 = np.zeros_like(F2)
dF3 = np.zeros_like(F3)
dFL = np.zeros_like(FL)

r_total = 0
r_total_plot = []
network_updates = 0
step = 0
err = 0
err_plot = []

###########
# init scene
reds_input = np.zeros((MEM_SZ, 2, N_REDS), dtype='int')
blues_input = np.zeros((MEM_SZ, 2, N_BLUES), dtype='int')
player_input = np.zeros((MEM_SZ, 2), dtype='int')
action_input = np.zeros(MEM_SZ, dtype='int')

r_output = np.zeros(MEM_SZ)
reds_output = np.zeros((MEM_SZ, 2, N_REDS), dtype='int')
blues_output = np.zeros((MEM_SZ, 2, N_BLUES), dtype='int')
player_output = np.zeros((MEM_SZ, 2), dtype='int')

reds = np.random.randint(0,MAX_LOC, size=(2,N_REDS))
blues = np.random.randint(0,MAX_LOC, size=(2,N_BLUES))
player = np.random.randint(0,MAX_LOC, size=2)

t_start = time.time()
while True:
	mem_loc  = step % MEM_SZ
	
	# copy current state
	reds_input[mem_loc] = copy.deepcopy(reds)
	blues_input[mem_loc] = copy.deepcopy(blues)
	player_input[mem_loc] = copy.deepcopy(player)
	
	# show blocks
	img = np.zeros((1,3,32,32),dtype='single')
	for b in range(N_REDS):
		img[0,0,SCALE*reds[0,b]:SCALE*(reds[0,b]+1), SCALE*reds[1,b]:SCALE*(reds[1,b]+1)] = 255
		img[0,2,SCALE*blues[0,b]:SCALE*(blues[0,b]+1), SCALE*blues[1,b]:SCALE*(blues[1,b]+1)] = 255
	img[0,1,SCALE*player[0]:SCALE*(player[0]+1), SCALE*player[1]:SCALE*(player[1]+1)] = 255
	
	# forward pass
	set_buffer(img, IMGS_PAD, gpu=GPU_CUR)
		
	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
	conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
	conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
	
	max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
	
	pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0])
	
	# choose action
	if (np.random.rand() <= EPS_GREED) or (step <= (MEM_SZ+RAND_PERIOD)):
		action = np.random.randint(4)
	else:
		action = np.argmax(pred)

	# perform action
	if action == 0 and player[0] != 0:
		player[0] -= 1
	elif action == 1 and (player[0]) != (MAX_LOC-1):
		player[0] += 1
	elif action == 2 and (player[1]) != 0:
		player[1] -= 1
	elif action == 3 and (player[1]) != (MAX_LOC-1):
		player[1] += 1

	
	# determine reward, choose new block locations
	r = 0

	# red collision, place new red block
	collision = np.nonzero((player[0] == reds[0]) * (player[1] == reds[1]))[0]
	if len(collision) >= 1:
		r = -1
		reds[:, collision] = np.random.randint(0,MAX_LOC, size=(2,1))

	# blue collision, place new blue block
	collision = np.nonzero((player[0] == blues[0]) * (player[1] == blues[1]))[0]
	if len(collision) >= 1:
		r = 1
		blues[:, collision] = np.random.randint(0,MAX_LOC, size=(2,1))

	r_total += r
	
	# copy current state
	reds_output[mem_loc] = copy.deepcopy(reds)
	blues_output[mem_loc] = copy.deepcopy(blues)
	player_output[mem_loc] = copy.deepcopy(player)
	r_output[mem_loc] = r
	action_input[mem_loc] = action
	
	if step == MEM_SZ:
		print 'beginning gradient computations'
	if step == (MEM_SZ+RAND_PERIOD):
		print 'beginning non-random actions'
	
	######################################
	# update gradient?
	if step >= MEM_SZ:
		trans = np.random.randint(MEM_SZ)
		
		# show blocks
		img_prev = np.zeros((1,3,32,32),dtype='single')
		img_cur = np.zeros((1,3,32,32),dtype='single')
		
		img_prev[0,1,SCALE*player_output[trans][0]:SCALE*(player_output[trans][0]+1), SCALE*player_output[trans][1]:SCALE*(player_output[trans][1]+1)] = 255
		img_cur[0,1,SCALE*player_input[trans][0]:SCALE*(player_input[trans][0]+1), SCALE*player_input[trans][1]:SCALE*(player_input[trans][1]+1)] = 255
		
		for b in range(N_REDS):
			img_prev[0,0,SCALE*reds_output[trans][0,b]:SCALE*(reds_output[trans][0,b]+1), SCALE*reds_output[trans][1,b]:SCALE*(reds_output[trans][1,b]+1)] = 255
			img_prev[0,2,SCALE*blues_output[trans][0,b]:SCALE*(blues_output[trans][0,b]+1), SCALE*blues_output[trans][1,b]:SCALE*(blues_output[trans][1,b]+1)] = 255
			
			img_cur[0,0,SCALE*reds_input[trans][0,b]:SCALE*(reds_input[trans][0,b]+1), SCALE*reds_input[trans][1,b]:SCALE*(reds_input[trans][1,b]+1)] = 255
			img_cur[0,2,SCALE*blues_input[trans][0,b]:SCALE*(blues_input[trans][0,b]+1), SCALE*blues_input[trans][1,b]:SCALE*(blues_input[trans][1,b]+1)] = 255
			
		
		set_buffer(img_cur, IMGS_PAD, gpu=GPU_CUR)
		set_buffer(img_prev, IMGS_PAD, gpu=GPU_PREV)
		
		# forward pass prev network
		conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_PREV)
		max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_PREV)
		conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_PREV)
		max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_PREV)
		conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_PREV)
		max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_PREV)
		
		# forward pass current network
		conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
		conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
		conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
		
		# compute target
		max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_PREV)
		pred_prev = np.einsum(FL_prev, range(4), max_output3, [4,1,2,3], [0])
		y = r_output[trans] + GAMMA * np.max(pred_prev)
		
		max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
		
		pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0])
		pred_m_Y = y - pred[action_input[trans]]
		
		err += pred_m_Y**2
		
		FL_pred = np.ascontiguousarray((FL[action_input[trans]] * pred_m_Y)[np.newaxis])
		
		set_buffer(FL_pred, FL_PRED, gpu=GPU_CUR)
		
		########### backprop
		max_pool_back_cudnn_buffers(MAX_OUTPUT3, FL_PRED, CONV_OUTPUT3, DPOOL3, gpu=GPU_CUR)
		conv_dfilter_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3, stream=3, gpu=GPU_CUR)
		conv_ddata_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3_DATA, gpu=GPU_CUR)
		max_pool_back_cudnn_buffers(MAX_OUTPUT2, DF3_DATA, CONV_OUTPUT2, DPOOL2, gpu=GPU_CUR)
		conv_ddata_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2_DATA, gpu=GPU_CUR)
		conv_dfilter_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2, stream=2, gpu=GPU_CUR)
		max_pool_back_cudnn_buffers(MAX_OUTPUT1, DF2_DATA, CONV_OUTPUT1, DPOOL1, gpu=GPU_CUR)
		conv_dfilter_buffers(F1_IND, IMGS_PAD, DPOOL1, DF1, stream=1, gpu=GPU_CUR)

		### return
		dFL[action_input[trans]] += max_output3[0]*pred_m_Y
		dF3 += return_buffer(DF3, stream=3, gpu=GPU_CUR)
		dF2 += return_buffer(DF2, stream=2, gpu=GPU_CUR)
		dF1 += return_buffer(DF1, stream=1, gpu=GPU_CUR)
		
		#### update filter weights
		if(step - MEM_SZ) % BATCH_SZ == 0:
			F1 -= dF1*EPS / BATCH_SZ
			F2 -= dF2*EPS / BATCH_SZ
			F3 -= dF3*EPS / BATCH_SZ
			FL -= dFL*EPS / BATCH_SZ
			
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
			set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_CUR)
			
			dF1 = np.zeros_like(dF1)
			dF2 = np.zeros_like(dF2)
			dF3 = np.zeros_like(dF3)
			dFL = np.zeros_like(dFL)
			
			err = 0
			
			network_updates += 1
			
			if network_updates % NETWORK_UPDATE == 0:
				print 'updating network'
				FL_prev = copy.deepcopy(FL)
				set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
				set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
				set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)
				set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_PREV)
				
	step += 1
	
	if step % 1000 == 0:
		r_total_plot.append(r_total)
		err_plot.append(err)
		savemat(file_name, {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, 'F1_init': F1_init, 'step': step, 'img': img, 'err_plot': err_plot})
		print step, err, r_total, network_updates, np.max(F1), time.time() - t_start, file_name
		t_start = time.time()
