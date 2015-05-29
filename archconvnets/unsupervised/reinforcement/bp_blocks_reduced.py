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

EPS_GREED_FINAL = .1
EPS_GREED_FINAL_TIME = 1000000
GAMMA = 0.99
BATCH_SZ = 1
NETWORK_UPDATE = 5000*32
EPS = 1e-3
MOM_WEIGHT = 0.95

SCALE = 4
MAX_LOC = 32/SCALE
N_REDS = 7
N_BLUES = 7

F1_scale = 1e-2
F2_scale = 1e-2
F3_scale = 1e-2
FL_scale = 1e-2

N = 32
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 4 # ...
s1 = 5

N_C = 4 # directions L, R, U, D

file_name = '/home/darren/reinforcement_batch1_sequential.mat'

max_output_sz3  = 5

GPU_CUR = 2
GPU_PREV = 3

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
reds = np.random.randint(0,MAX_LOC, size=(2,N_REDS))
blues = np.random.randint(0,MAX_LOC, size=(2,N_BLUES))
player = np.random.randint(0,MAX_LOC, size=2)

# show blocks
img = np.zeros((1,3,32,32),dtype='single')
for b in range(N_REDS):
	img[0,0,SCALE*reds[0,b]:SCALE*(reds[0,b]+1), SCALE*reds[1,b]:SCALE*(reds[1,b]+1)] = 255
	img[0,2,SCALE*blues[0,b]:SCALE*(blues[0,b]+1), SCALE*blues[1,b]:SCALE*(blues[1,b]+1)] = 255
img[0,1,SCALE*player[0]:SCALE*(player[0]+1), SCALE*player[1]:SCALE*(player[1]+1)] = 255

t_start = time.time()
while True:
	# copy current state
	reds_input = copy.deepcopy(reds)
	blues_input = copy.deepcopy(blues)
	player_input = copy.deepcopy(player)

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
	CHANCE_RAND = np.min((1 - ((1-EPS_GREED_FINAL)/EPS_GREED_FINAL_TIME)*step, 1))
	if np.random.rand() <= CHANCE_RAND:
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

	########### forward pass on next img w/ prev network
	# show blocks
	img = np.zeros((1,3,32,32),dtype='single')
	for b in range(N_REDS):
		img[0,0,SCALE*reds[0,b]:SCALE*(reds[0,b]+1), SCALE*reds[1,b]:SCALE*(reds[1,b]+1)] = 255
		img[0,2,SCALE*blues[0,b]:SCALE*(blues[0,b]+1), SCALE*blues[1,b]:SCALE*(blues[1,b]+1)] = 255
	img[0,1,SCALE*player[0]:SCALE*(player[0]+1), SCALE*player[1]:SCALE*(player[1]+1)] = 255
	
	
	# forward pass prev network
	set_buffer(img, IMGS_PAD, gpu=GPU_PREV)

	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_PREV)
	conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_PREV)
	conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_PREV)

	# compute target
	max_output3_prev = return_buffer(MAX_OUTPUT3, gpu=GPU_PREV)
	pred_prev = np.einsum(FL_prev, range(4), max_output3_prev, [4,1,2,3], [0])
	y_output = r + GAMMA * np.max(pred_prev)
	
	######################################
	# gradient
	pred_m_Y = y_output - pred[action]
	
	err += pred_m_Y**2
	
	FL_pred = np.ascontiguousarray((FL[action] * pred_m_Y)[np.newaxis])
	
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
	dFL[action] += max_output3[0]*pred_m_Y
	dF3 += return_buffer(DF3, stream=3, gpu=GPU_CUR)
	dF2 += return_buffer(DF2, stream=2, gpu=GPU_CUR)
	dF1 += return_buffer(DF1, stream=1, gpu=GPU_CUR)
	
	#### update filter weights
	if step % BATCH_SZ == 0:
		F1 += dF1*EPS / BATCH_SZ
		F2 += dF2*EPS / BATCH_SZ
		F3 += dF3*EPS / BATCH_SZ
		FL += dFL*EPS / BATCH_SZ
		
		set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
		set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
		set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
		set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_CUR)
		
		dF1 = np.zeros_like(dF1)
		dF2 = np.zeros_like(dF2)
		dF3 = np.zeros_like(dF3)
		dFL = np.zeros_like(dFL)
		
		network_updates += 1
		
		if network_updates % NETWORK_UPDATE == 0:
			print 'updating network'
			FL_prev = copy.deepcopy(FL)
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)
			set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_PREV)
				
	step += 1
	
	if step % 4000 == 0:
		r_total_plot.append(r_total)
		err_plot.append(err)
		
		savemat(file_name, {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, 'F1_init': F1_init, 'step': step, 'img': img, 'err_plot': err_plot})
		print 'step:', step, 'err:',err, 'r:',r_total, 'updates:',network_updates, 'eps:', CHANCE_RAND, 'F1:', np.max(F1), 't:',time.time() - t_start, file_name
		
		err = 0
		r_total = 0
		
		t_start = time.time()
