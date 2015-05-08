from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy

RAND_PERIOD = 0
MEM_SZ = 100000
EPS_GREED_FINAL = .1
EPS_GREED_FINAL_TIME = 2*1000000
GAMMA = 0.99
BATCH_SZ = 32
NETWORK_UPDATE = 10000
EPS = 5e-3
MOM_WEIGHT = 0.95
SAVE_FREQ = 5000

SCALE = 4
MAX_LOC = 32 - SCALE

F1_scale = 1e-2
F2_scale = 1e-2
F3_scale = 1e-2
FL_scale = 1e-2
CEC_SCALE = 1e-3
FCF_SCALE = 1e-4

N = 32
n1 = N # L1 filters
n2 = N # ...
n3 = N
n4 = N+1

s3 = 3 # L1 filter size (px)
s2 = 4 # ...
s1 = 5

N_C = 4 # directions L, R, U, D

file_name = '/home/darren/reinforcement_blocks_moving2.mat'

PLAYER_MOV_RATE = 3
RED_MOV_RATE = 1
BLUE_MOV_RATE = 1

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

FCm = np.single(np.random.normal(scale=FL_scale, size=(n4, n3, max_output_sz3, max_output_sz3)))
FCi = np.single(np.random.normal(scale=FL_scale, size=(n4, n3, max_output_sz3, max_output_sz3)))
FCo = np.single(np.random.normal(scale=FL_scale, size=(n4, n3, max_output_sz3, max_output_sz3)))
FCf = np.single(np.random.normal(scale=FCF_SCALE, size=(n4, n3, max_output_sz3, max_output_sz3)))
CEC = np.single(np.random.normal(scale=CEC_SCALE, size=(n4)))

FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n4)))
FL_bypass = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

FCm_prev = copy.deepcopy(FCm)
FCi_prev = copy.deepcopy(FCi)
FCo_prev = copy.deepcopy(FCo)
FCf_prev = copy.deepcopy(FCf)
CEC_prev = copy.deepcopy(CEC)

FL_prev = copy.deepcopy(FL)
FL_bypass_prev = copy.deepcopy(FL_bypass)

set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)

set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)

F1_init = copy.deepcopy(F1)

dF1 = np.zeros_like(F1)
dF2 = np.zeros_like(F2)
dF3 = np.zeros_like(F3)
dFL = np.zeros_like(FL)
dFL_bypass = np.zeros_like(FL_bypass)

dFCm = np.zeros_like(FCm)
dFCi = np.zeros_like(FCi)
dFCo = np.zeros_like(FCo)
dFCf = np.zeros_like(FCf)

dF1_mom = np.zeros_like(F1)
dF2_mom = np.zeros_like(F2)
dF3_mom = np.zeros_like(F3)
dFL_mom = np.zeros_like(FL)
dFL_bypass_mom = np.zeros_like(FL_bypass)

dFCm_mom = np.zeros_like(FCm)
dFCi_mom = np.zeros_like(FCi)
dFCo_mom = np.zeros_like(FCo)
dFCf_mom = np.zeros_like(FCf)

r_total = 0
r_total_plot = []
network_updates = 0
step = 0
err = 0
err_plot = []

###########
# init scene
reds_input = np.zeros((MEM_SZ, 2), dtype='single')
blues_input = np.zeros((MEM_SZ, 2), dtype='single')
player_input = np.zeros((MEM_SZ, 2), dtype='int')
action_input = np.zeros(MEM_SZ, dtype='int')
CEC_input = np.zeros((MEM_SZ, n4), dtype='single')
blue_direction_input = np.zeros((MEM_SZ, 2), dtype='single')
red_direction_input = np.zeros((MEM_SZ, 2), dtype='single')

r_output = np.zeros(MEM_SZ)
y_outputs = np.zeros(MEM_SZ)
y_network_ver = -np.ones(MEM_SZ)
reds_output = np.zeros((MEM_SZ, 2), dtype='single')
blues_output = np.zeros((MEM_SZ, 2), dtype='single')
player_output = np.zeros((MEM_SZ, 2), dtype='int')
CEC_output = np.zeros((MEM_SZ, n4), dtype='single')
blue_direction_output = np.zeros((MEM_SZ, 2), dtype='single')
red_direction_output = np.zeros((MEM_SZ, 2), dtype='single')

red_direction = np.random.randint(2)
blue_direction = np.random.randint(2)

reds = np.zeros(2, dtype='single')
blues = np.zeros(2, dtype='single')

reds[0] = (32+SCALE) * np.random.random() - SCALE
blues[1] = (32+SCALE) * np.random.random() - SCALE

reds[1] = (32+SCALE) * np.random.random() - SCALE
blues[0] = (32+SCALE) * np.random.random() - SCALE

player = np.random.randint(0,MAX_LOC, size=2)

imgs_recent = np.zeros((SAVE_FREQ, 3, 32, 32), dtype='single')
imgs_recent_key = np.zeros((SAVE_FREQ, 3, 32, 32), dtype='single')

imgs_mean = np.zeros((3,32,32))
imgs_mean_key = np.zeros((3,32,32))

t_start = time.time()
while True:
	mem_loc  = step % MEM_SZ
	
	# copy current state
	reds_input[mem_loc] = copy.deepcopy(reds)
	blues_input[mem_loc] = copy.deepcopy(blues)
	player_input[mem_loc] = copy.deepcopy(player)
	CEC_input[mem_loc] = copy.deepcopy(CEC)
	red_direction_input[mem_loc] = copy.deepcopy(red_direction)
	blue_direction_input[mem_loc] = copy.deepcopy(blue_direction)
	y_network_ver[mem_loc] = -1
	
	# show blocks
	img = np.zeros((1,3,32,32),dtype='single')
	img[0,2,np.max((np.round(reds[0]),0)):np.round(reds[0]+SCALE), np.max((np.round(reds[1]),0)):np.round(reds[1]+SCALE)] = 255
	img[0,2,np.max((np.round(blues[0]),0)):np.round(blues[0]+SCALE), np.max((np.round(blues[1]),0)):np.round(blues[1]+SCALE)] = 255
	img[0,1,player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] = 255
	
	img_key = copy.deepcopy(img)
	img_key[0,0,np.max((np.round(blues[0]),0)):np.round(blues[0]+SCALE), np.max((np.round(blues[1]),0)):np.round(blues[1]+SCALE)] = 255
	
	# forward pass
	set_buffer(img, IMGS_PAD, gpu=GPU_CUR)
		
	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
	conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
	conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
	
	max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
	
	FCm_output = np.einsum(FCm, range(4), max_output3, [4, 1,2,3], [4, 0])
	FCi_output = np.einsum(FCi, range(4), max_output3, [4, 1,2,3], [4, 0])
	FCo_output = np.einsum(FCo, range(4), max_output3, [4, 1,2,3], [4, 0])
	FCf_output = np.einsum(FCf, range(4), max_output3, [4, 1,2,3], [4, 0])
	
	FC_output = FCo_output*(CEC*FCf_output + FCi_output*FCm_output)
	
	#CEC = CEC*FCf_output + FCi_output*FCm_output
	
	# choose action
	CHANCE_RAND = np.max((1 - ((1-EPS_GREED_FINAL)/EPS_GREED_FINAL_TIME)*(step - MEM_SZ), EPS_GREED_FINAL))
	if np.random.rand() <= CHANCE_RAND:
		action = np.random.randint(4)
	else:
		pred = np.einsum(FL, [0,1], FC_output, [2, 1], [0])
		pred += np.einsum(FL_bypass, range(4), max_output3, [4,1,2,3], [0])
		
		action = np.argmax(pred)

	# perform action
	if action == 0 and player[0] >= PLAYER_MOV_RATE:
		player[0] -= PLAYER_MOV_RATE
	elif action == 1 and (player[0]) <= (MAX_LOC-1):
		player[0] += PLAYER_MOV_RATE
	elif action == 2 and (player[1]) >= PLAYER_MOV_RATE:
		player[1] -= PLAYER_MOV_RATE
	elif action == 3 and (player[1]) <= (MAX_LOC-1):
		player[1] += PLAYER_MOV_RATE

	
	# determine reward, choose new block locations
	r = 0

	# red collision, place new red block
	if (player[0] >= reds[0]) * (player[0] <= (reds[0] + SCALE)) * (player[1] >= reds[1]) * (player[1] <= (reds[1] + SCALE)) + \
		(player[0] <= reds[0]) * ((player[0] + SCALE) >= reds[0]) * (player[1] >= reds[1]) * (player[1] <= (reds[1] + SCALE)) + \
		(player[0] >= reds[0]) * (player[0] <= (reds[0] + SCALE)) * (player[1] <= reds[1]) * ((player[1] + SCALE) >= reds[1]) + \
		(player[0] <= reds[0]) * ((player[0] + SCALE) >= reds[0]) * (player[1] <= reds[1]) * ((player[1] + SCALE) >= reds[1]):
			r = -1
			red_direction = np.random.randint(2)
			
			reds[0] = (32+SCALE) * np.random.random() - SCALE
			reds[1] = (32+SCALE) * np.random.random() - SCALE

	# blue collision, place new blue block
	if (player[0] >= blues[0]) * (player[0] <= (blues[0] + SCALE)) * (player[1] >= blues[1]) * (player[1] <= (blues[1] + SCALE)) + \
		(player[0] <= blues[0]) * ((player[0] + SCALE) >= blues[0]) * (player[1] >= blues[1]) * (player[1] <= (blues[1] + SCALE)) + \
		(player[0] >= blues[0]) * (player[0] <= (blues[0] + SCALE)) * (player[1] <= blues[1]) * ((player[1] + SCALE) >= blues[1]) + \
		(player[0] <= blues[0]) * ((player[0] + SCALE) >= blues[0]) * (player[1] <= blues[1]) * ((player[1] + SCALE) >= blues[1]):
			r = 1
			blue_direction = np.random.randint(2)
			
			blues[1] = (32+SCALE) * np.random.random() - SCALE
			blues[0] = (32+SCALE) * np.random.random() - SCALE

	r_total += r
	
	# move blocks
	reds[0] -= RED_MOV_RATE * (2*red_direction - 1)
	blues[1] -= BLUE_MOV_RATE * (2*blue_direction - 1)
	
	# have any blocks moved off screen?
	if reds[0] < -SCALE or reds[0] > 32:
		red_direction = np.random.randint(2)
			
		reds[0] = (32+SCALE) * np.random.random() - SCALE
		reds[1] = (32+SCALE) * np.random.random() - SCALE
		
	if blues[1] < -SCALE or blues[1] > 32:
		blue_direction = np.random.randint(2)
			
		blues[1] = (32+SCALE) * np.random.random() - SCALE
		blues[0] = (32+SCALE) * np.random.random() - SCALE
	
	# copy current state
	reds_output[mem_loc] = copy.deepcopy(reds)
	blues_output[mem_loc] = copy.deepcopy(blues)
	player_output[mem_loc] = copy.deepcopy(player)
	r_output[mem_loc] = r
	CEC_output[mem_loc] = copy.deepcopy(CEC)
	red_direction_output[mem_loc] = copy.deepcopy(red_direction)
	blue_direction_output[mem_loc] = copy.deepcopy(blue_direction)
	action_input[mem_loc] = action
	
	# debug/visualization
	save_loc = step % SAVE_FREQ
	
	imgs_recent[save_loc] = copy.deepcopy(img[0])
	imgs_recent_key[save_loc] = copy.deepcopy(img_key[0])
	imgs_mean_key += img_key[0]
	imgs_mean += img[0]
	
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
		
		img_prev[0,1,player_output[trans][0]:(player_output[trans][0]+SCALE), player_output[trans][1]:(player_output[trans][1]+SCALE)] = 255
		img_cur[0,1,player_input[trans][0]:(player_input[trans][0]+SCALE), player_input[trans][1]:(player_input[trans][1]+SCALE)] = 255
		
		img_prev[0,2,np.max((np.round(reds_output[trans][0]),0)):np.round(reds_output[trans][0]+SCALE), np.max((np.round(reds_output[trans][1]),0)):np.round(reds_output[trans][1]+SCALE)] = 255
		img_prev[0,2,np.max((np.round(blues_output[trans][0]),0)):np.round(blues_output[trans][0]+SCALE), np.max((np.round(blues_output[trans][1]),0)):np.round(blues_output[trans][1]+SCALE)] = 255
		
		img_cur[0,2,np.max((np.round(reds_input[trans][0]),0)):np.round(reds_input[trans][0]+SCALE), np.max((np.round(reds_input[trans][1]),0)):np.round(reds_input[trans][1]+SCALE)] = 255
		img_cur[0,2,np.max((np.round(blues_input[trans][0]),0)):np.round(blues_input[trans][0]+SCALE), np.max((np.round(blues_input[trans][1]),0)):np.round(blues_input[trans][1]+SCALE)] = 255
			
		
		set_buffer(img_cur, IMGS_PAD, gpu=GPU_CUR)
		
		# forward pass current network
		conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
		conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
		conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
		
		# forward pass prev network
		if y_network_ver[trans] != (network_updates % NETWORK_UPDATE):
			set_buffer(img_prev, IMGS_PAD, gpu=GPU_PREV)
		
			conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_PREV)
			max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_PREV)
			conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_PREV)
			max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_PREV)
			conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_PREV)
			max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_PREV)
		
			# compute target
			max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_PREV)
			
			FCm_output = np.einsum(FCm_prev, range(4), max_output3, [4, 1,2,3], [4, 0])
			FCi_output = np.einsum(FCi_prev, range(4), max_output3, [4, 1,2,3], [4, 0])
			FCo_output = np.einsum(FCo_prev, range(4), max_output3, [4, 1,2,3], [4, 0])
			FCf_output = np.einsum(FCf_prev, range(4), max_output3, [4, 1,2,3], [4, 0])
			
			FC_output = FCo_output*(CEC_output[trans]*FCf_output + FCi_output*FCm_output)
			
			pred_prev = np.einsum(FL_prev, [0,1], FC_output, [2, 1], [0])
			pred_prev += np.einsum(FL_bypass_prev, range(4), max_output3, [4,1,2,3], [0])
			
			y_outputs[trans] = r_output[trans] + GAMMA * np.max(pred_prev)
			y_network_ver[trans] = network_updates % NETWORK_UPDATE
		
		max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
		
		FCm_output = np.einsum(FCm, range(4), max_output3, [4, 1,2,3], [4, 0])
		FCi_output = np.einsum(FCi, range(4), max_output3, [4, 1,2,3], [4, 0])
		FCo_output = np.einsum(FCo, range(4), max_output3, [4, 1,2,3], [4, 0])
		FCf_output = np.einsum(FCf, range(4), max_output3, [4, 1,2,3], [4, 0])
		
		FC_output = FCo_output*(CEC_input[trans]*FCf_output + FCi_output*FCm_output)
		
		pred = np.einsum(FL, [0,1], FC_output, [2, 1], [0])
		pred += np.einsum(FL_bypass, range(4), max_output3, [4,1,2,3], [0])
		
		pred_m_Y = y_outputs[trans] - pred[action_input[trans]]
		
		err += pred_m_Y**2
		
		FL_pred = (FL[action_input[trans]] * pred_m_Y)[np.newaxis]
		
		FLFC_pred = np.einsum(FL_pred*(CEC_input[trans]*FCf_output + FCi_output*FCm_output), [0,1], FCo, [1,2,3,4], [0, 2,3,4])
		FLFC_pred += np.einsum(FCo_output*(CEC_input[trans]*FL_pred), [0,1], FCf, [1,2,3,4], [0, 2,3,4])
		FLFC_pred += np.einsum(FCo_output*(FL_pred*FCm_output), [0,1], FCi, [1,2,3,4], [0, 2,3,4])
		FLFC_pred += np.einsum(FCo_output*(FL_pred*FCi_output), [0,1], FCm, [1,2,3,4], [0, 2,3,4])
		
		FLFC_pred += np.ascontiguousarray((FL_bypass[action_input[trans]] * pred_m_Y)[np.newaxis])
		
		set_buffer(FLFC_pred, FL_PRED, gpu=GPU_CUR)
		
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
		dFL[action_input[trans]] += FC_output[0]*pred_m_Y
		dFL_bypass[action_input[trans]] += max_output3[0]*pred_m_Y
		
		dFCf += np.einsum(max_output3, range(4), FL_pred*FCo_output*CEC_input[trans], [0,4], [4,1,2,3])
		dFCo += np.einsum(max_output3, range(4), FL_pred*(CEC_input[trans]*FCf_output + FCi_output*FCm_output), [0,4], [4,1,2,3])
		dFCm += np.einsum(max_output3, range(4), FCi_output*FL_pred*FCo_output, [0,4], [4,1,2,3])
		dFCi += np.einsum(max_output3, range(4), FCm_output*FL_pred*FCo_output, [0,4], [4,1,2,3])
		
		dF3 += return_buffer(DF3, stream=3, gpu=GPU_CUR)
		dF2 += return_buffer(DF2, stream=2, gpu=GPU_CUR)
		dF1 += return_buffer(DF1, stream=1, gpu=GPU_CUR)
		
		#### update filter weights
		if(step - MEM_SZ) % BATCH_SZ == 0:
			F1 += (dF1 + MOM_WEIGHT*dF1_mom)*EPS / BATCH_SZ
			F2 += (dF2 + MOM_WEIGHT*dF2_mom)*EPS / BATCH_SZ
			F3 += (dF3 + MOM_WEIGHT*dF3_mom)*EPS / BATCH_SZ
			FL += (dFL + MOM_WEIGHT*dFL_mom)*EPS / BATCH_SZ
			FL_bypass += (dFL_bypass + MOM_WEIGHT*dFL_bypass_mom)*EPS / BATCH_SZ
			
			dFCf += (dFCf + MOM_WEIGHT*dFCf_mom)*EPS / BATCH_SZ
			dFCo += (dFCo + MOM_WEIGHT*dFCo_mom)*EPS / BATCH_SZ
			dFCm += (dFCm + MOM_WEIGHT*dFCm_mom)*EPS / BATCH_SZ
			dFCi += (dFCi + MOM_WEIGHT*dFCi_mom)*EPS / BATCH_SZ
			
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
			
			dF1_mom = copy.deepcopy(dF1)
			dF2_mom = copy.deepcopy(dF2)
			dF3_mom = copy.deepcopy(dF3)
			dFL_mom = copy.deepcopy(dFL)
			dFL_bypass_mom = copy.deepcopy(dFL_bypass)
			
			dFCf_mom = copy.deepcopy(dFCf)
			dFCo_mom = copy.deepcopy(dFCo)
			dFCm_mom = copy.deepcopy(dFCm)
			dFCi_mom = copy.deepcopy(dFCi)
			
			dF1 = np.zeros_like(dF1)
			dF2 = np.zeros_like(dF2)
			dF3 = np.zeros_like(dF3)
			dFL = np.zeros_like(dFL)
			dFL_bypass = np.zeros_like(dFL_bypass)
			
			dFCf = np.zeros_like(dFCf)
			dFCo = np.zeros_like(dFCo)
			dFCm = np.zeros_like(dFCm)
			dFCi = np.zeros_like(dFCi)
			
			network_updates += 1
			
			if network_updates % NETWORK_UPDATE == 0:
				print 'updating network'
				FL_prev = copy.deepcopy(FL)
				FL_bypass_prev = copy.deepcopy(FL_bypass)
				
				FCf_prev = copy.deepcopy(FCf)
				FCo_prev = copy.deepcopy(FCo)
				FCm_prev = copy.deepcopy(FCm)
				FCi_prev = copy.deepcopy(FCi)
				
				set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
				set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
				set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)
				
	step += 1
	
	if step % SAVE_FREQ == 0:
		r_total_plot.append(r_total)
		err_plot.append(err)
		
		savemat(file_name, {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, 'F1_init': F1_init, 'step': step, 'img': img, 'err_plot': err_plot, 'CEC':CEC, 'CEC_input': CEC_input, 'imgs_mean': imgs_mean, 'imgs_mean_key': imgs_mean_key, \
				'CEC_output':CEC_output, 'imgs_recent': imgs_recent, 'imgs_recent_key': imgs_recent_key})
		print 'step:', step, 'err:',err, 'r:',r_total, 'updates:',network_updates, 'eps:', CHANCE_RAND, 'F1:', np.max(F1), 't:',time.time() - t_start, file_name, np.min(CEC), np.max(CEC)
		
		err = 0
		r_total = 0
		
		t_start = time.time()
