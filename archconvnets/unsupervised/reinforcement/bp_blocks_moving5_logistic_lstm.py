#online?
from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy

EPS_GREED_FINAL = .1
EPS_GREED_FINAL_TIME = 2000000
GAMMA = 0.99
BATCH_SZ = 1
NETWORK_UPDATE = 10000
EPS = 2e-3
SAVE_FREQ = 5000

SCALE = 4
MAX_LOC = 32 - SCALE

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = .01
FL2_scale = .01
FL3_scale = .02
CEC_SCALE = 0.001

N = 32
n1 = N # L1 filters
n2 = N # ...
n3 = N
n1m = 4*128#*4*2
n2m = 4*128 +2#+ 2#*4*2
n3m = 128 +1#+ 3#*4*2

s3 = 3 # L1 filter size (px)
s2 = 4 # ...
s1 = 5

N_C = 4 # directions L, R, U, D

file_name = '/home/darren/reinforcement_blocks_moving_CEC_FLm.mat'

PLAYER_MOV_RATE = 3
RED_MOV_RATE = 1
BLUE_MOV_RATE = 1

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

mem_contrib = 0; curr_contrib = 0

#np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))

## l1
FCm = np.single(np.random.normal(scale=FL_scale, size=(n1m, n3, max_output_sz3, max_output_sz3)))
FCi = np.single(np.random.normal(scale=FL_scale, size=(n1m, n3, max_output_sz3, max_output_sz3)))
FCo = np.single(np.random.normal(scale=FL_scale, size=(n1m, n3, max_output_sz3, max_output_sz3)))
FCf = np.single(np.random.normal(scale=FL_scale, size=(n1m, n3, max_output_sz3, max_output_sz3)))

Bm = np.single(np.random.normal(scale=FL_scale, size=n1m))
Bi = np.single(np.random.normal(scale=FL_scale, size=n1m))
Bo = np.single(np.random.normal(scale=FL_scale, size=n1m))
Bf = np.single(np.random.normal(scale=FL_scale, size=n1m))
CEC = np.single(np.random.normal(scale=CEC_SCALE, size=(n1m)))

## l2
FC2f = np.single(np.random.normal(scale=FL2_scale, size=(n2m, n1m)))
FC2o = np.single(np.random.normal(scale=FL2_scale, size=(n2m, n1m)))
FC2i = np.single(np.random.normal(scale=FL2_scale, size=(n2m, n1m)))
FC2m = np.single(np.random.normal(scale=FL2_scale, size=(n2m, n1m)))

B2m = np.single(np.random.normal(scale=FL2_scale, size=n2m))
B2i = np.single(np.random.normal(scale=FL2_scale, size=n2m))
B2o = np.single(np.random.normal(scale=FL2_scale, size=n2m))
B2f = np.single(np.random.normal(scale=FL2_scale, size=n2m))
CEC2 = np.single(np.random.normal(scale=CEC_SCALE, size=n2m))

### l3
FC3f = np.single(np.random.normal(scale=FL3_scale, size=(n3m, n2m)))
FC3o = np.single(np.random.normal(scale=FL3_scale, size=(n3m, n2m)))
FC3i = np.single(np.random.normal(scale=FL3_scale, size=(n3m, n2m)))
FC3m = np.single(np.random.normal(scale=FL3_scale, size=(n3m, n2m)))

B3m = np.single(np.random.normal(scale=FL3_scale, size=n3m))
B3i = np.single(np.random.normal(scale=FL3_scale, size=n3m))
B3o = np.single(np.random.normal(scale=FL3_scale, size=n3m))
B3f = np.single(np.random.normal(scale=FL3_scale, size=n3m))
CEC3 = np.single(np.random.normal(scale=CEC_SCALE, size=n3m))

FL = np.single(np.random.normal(scale=5, size=(4,n3m)))

CEC3_dFC3m = np.zeros_like(CEC3)
CEC3_dFC3i = np.zeros_like(CEC3)
CEC3_dFC3f = np.zeros_like(CEC3)

CEC2_dFC2m = np.zeros_like(CEC2)
CEC2_dFC2i = np.zeros_like(CEC2)
CEC2_dFC2f = np.zeros_like(CEC2)

CEC_dFCm = np.zeros_like(CEC)
CEC_dFCi = np.zeros_like(CEC)
CEC_dFCf = np.zeros_like(CEC)

###
CEC_prev = copy.deepcopy(CEC)
CEC2_prev = copy.deepcopy(CEC2)
CEC3_prev = copy.deepcopy(CEC3)

FCm_prev = copy.deepcopy(FCm)
FCi_prev = copy.deepcopy(FCi)
FCo_prev = copy.deepcopy(FCo)
FCf_prev = copy.deepcopy(FCf)

Bm_prev = copy.deepcopy(Bm)
Bi_prev = copy.deepcopy(Bi)
Bo_prev = copy.deepcopy(Bo)
Bf_prev = copy.deepcopy(Bf)

FC2m_prev = copy.deepcopy(FC2m)
FC2i_prev = copy.deepcopy(FC2i)
FC2o_prev = copy.deepcopy(FC2o)
FC2f_prev = copy.deepcopy(FC2f)

B2m_prev = copy.deepcopy(B2m)
B2i_prev = copy.deepcopy(B2i)
B2o_prev = copy.deepcopy(B2o)
B2f_prev = copy.deepcopy(B2f)

FC3m_prev = copy.deepcopy(FC3m)
FC3i_prev = copy.deepcopy(FC3i)
FC3o_prev = copy.deepcopy(FC3o)
FC3f_prev = copy.deepcopy(FC3f)

B3m_prev = copy.deepcopy(B3m)
B3i_prev = copy.deepcopy(B3i)
B3o_prev = copy.deepcopy(B3o)
B3f_prev = copy.deepcopy(B3f)

FL_prev = copy.deepcopy(FL)

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

dFCm = np.zeros_like(FCm)
dFCi = np.zeros_like(FCi)
dFCo = np.zeros_like(FCo)
dFCf = np.zeros_like(FCf)

dBm = np.zeros_like(Bm)
dBi = np.zeros_like(Bi)
dBo = np.zeros_like(Bo)
dBf = np.zeros_like(Bf)

dFC2m = np.zeros_like(FC2m)
dFC2i = np.zeros_like(FC2i)
dFC2o = np.zeros_like(FC2o)
dFC2f = np.zeros_like(FC2f)

dB2m = np.zeros_like(B2m)
dB2i = np.zeros_like(B2i)
dB2o = np.zeros_like(B2o)
dB2f = np.zeros_like(B2f)

dFC3m = np.zeros_like(FC3m)
dFC3i = np.zeros_like(FC3i)
dFC3o = np.zeros_like(FC3o)
dFC3f = np.zeros_like(FC3f)

dB3m = np.zeros_like(B3m)
dB3i = np.zeros_like(B3i)
dB3o = np.zeros_like(B3o)
dB3f = np.zeros_like(B3f)


r_total = 0
r_total_plot = []
network_updates = 0
step = 0
err = 0
err_plot = []

###########
# init scene
reds = np.zeros(2, dtype='single')
reward_phase = 1

show_transition = 255

red_axis = np.random.randint(2)
red_direction = np.random.randint(2)

if red_direction == 1:
	reds[red_axis] = 32
else:
	reds[red_axis] = -SCALE

reds[1 - red_axis] = (32+SCALE) * np.random.random() - SCALE

player = np.random.randint(0,MAX_LOC, size=2)

imgs_recent = np.zeros((SAVE_FREQ, 3, 32, 32), dtype='single')
imgs_recent_key = np.zeros((SAVE_FREQ, 3, 32, 32), dtype='single')
CEC_recent = np.zeros((SAVE_FREQ, n1m), dtype='single')
action_recent = np.zeros(SAVE_FREQ, dtype='int')
r_recent = np.zeros(SAVE_FREQ, dtype='single')

imgs_mean_red = np.zeros((32,32))
imgs_mean_player = np.zeros((32,32))

t_start = time.time()

######## mean img
N_MEAN = 500000
img_mean = np.zeros((1,3,32,32),dtype='single')
for sample in range(N_MEAN):
	red_axis = np.random.randint(2)
	red_direction = np.random.randint(2)

	if red_direction == 1:
		reds[red_axis] = 32
	else:
		reds[red_axis] = -SCALE

	reds[1 - red_axis] = (32+SCALE) * np.random.random() - SCALE

	player = np.random.randint(0,MAX_LOC, size=2)
	img_mean[0,2,np.max((np.round(reds[0]),0)):np.round(reds[0]+SCALE), np.max((np.round(reds[1]),0)):np.round(reds[1]+SCALE)] += 255
	img_mean[0,1,player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] += 255
img_mean /= N_MEAN

# show blocks
img = np.zeros((1,3,32,32),dtype='single')
img[0,2,np.max((np.round(reds[0]),0)):np.round(reds[0]+SCALE), np.max((np.round(reds[1]),0)):np.round(reds[1]+SCALE)] = 255
img[0,1,player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] = 255
img[0,reward_phase+1,player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] = show_transition

show_transition = 0

while True:
	# debug/visualizations
	imgs_mean_red[np.max((np.round(reds[0]),0)):np.round(reds[0]+SCALE), np.max((np.round(reds[1]),0)):np.round(reds[1]+SCALE)] += 1
	imgs_mean_player[player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] += 1
	
	# forward pass
	set_buffer(img - img_mean, IMGS_PAD, gpu=GPU_CUR)
		
	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
	conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_CUR)
	conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_CUR)
	
	max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_CUR)
	
	FCm_output_pre = np.einsum(FCm, range(4), max_output3, [4, 1,2,3], [4, 0]) + Bm
	FCi_output_pre = np.einsum(FCi, range(4), max_output3, [4, 1,2,3], [4, 0]) + Bi
	FCo_output_pre = np.einsum(FCo, range(4), max_output3, [4, 1,2,3], [4, 0]) + Bo
	FCf_output_pre = np.einsum(FCf, range(4), max_output3, [4, 1,2,3], [4, 0]) + Bf
	
	FCf_output = 1 / (1 + np.exp(-FCf_output_pre))
	FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
	FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
	FCm_output = FCm_output_pre
	
	FC_output = FCo_output * (FCf_output * CEC + FCi_output * FCm_output)
	
	CEC_kept = CEC*FCf_output
	CEC_new = FCi_output*FCm_output
	
	CEC = CEC*FCf_output + FCi_output*FCm_output
	
	FC2f_output_pre = np.einsum(FC2f, [0,1], FC_output, [2,1], [2,0]) + B2f
	FC2o_output_pre = np.einsum(FC2o, [0,1], FC_output, [2,1], [2,0]) + B2o
	FC2i_output_pre = np.einsum(FC2i, [0,1], FC_output, [2,1], [2,0]) + B2i
	FC2m_output_pre = np.einsum(FC2m, [0,1], FC_output, [2,1], [2,0]) + B2m
	
	FC2f_output = 1 / (1 + np.exp(-FC2f_output_pre))
	FC2o_output = 1 / (1 + np.exp(-FC2o_output_pre))
	FC2i_output = 1 / (1 + np.exp(-FC2i_output_pre))
	FC2m_output = FC2m_output_pre
	
	FC2_output = FC2o_output * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	CEC2_kept = CEC2*FC2f_output
	CEC2_new = FC2i_output*FC2m_output
	
	CEC2 = CEC2*FC2f_output + FC2i_output*FC2m_output
	
	FC3f_output_pre = np.einsum(FC3f, [0,1], FC2_output, [2,1], [2,0]) + B3f
	FC3o_output_pre = np.einsum(FC3o, [0,1], FC2_output, [2,1], [2,0]) + B3o
	FC3i_output_pre = np.einsum(FC3i, [0,1], FC2_output, [2,1], [2,0]) + B3i
	FC3m_output_pre = np.einsum(FC3m, [0,1], FC2_output, [2,1], [2,0]) + B3m
	
	FC3f_output = 1 / (1 + np.exp(-FC3f_output_pre))
	FC3o_output = 1 / (1 + np.exp(-FC3o_output_pre))
	FC3i_output = 1 / (1 + np.exp(-FC3i_output_pre))
	FC3m_output = FC3m_output_pre
	
	FC3_output = FC3o_output * (FC3f_output * CEC3 + FC3i_output * FC3m_output)
	
	pred = np.einsum(FL, [0,1], FC3_output, [2, 1], [0])
	
	############### reverse pointwise
	FC3f_output_rev = np.exp(FC3f_output_pre)/((np.exp(FC3f_output_pre) + 1)**2)
	FC3o_output_rev = np.exp(FC3o_output_pre)/((np.exp(FC3o_output_pre) + 1)**2)
	FC3i_output_rev = np.exp(FC3i_output_pre)/((np.exp(FC3i_output_pre) + 1)**2)
	FC3m_output_rev = 1
	
	FC2f_output_rev = np.exp(FC2f_output_pre)/((np.exp(FC2f_output_pre) + 1)**2)
	FC2o_output_rev = np.exp(FC2o_output_pre)/((np.exp(FC2o_output_pre) + 1)**2)
	FC2i_output_rev = np.exp(FC2i_output_pre)/((np.exp(FC2i_output_pre) + 1)**2)
	FC2m_output_rev = 1
	
	FCf_output_rev = np.exp(FCf_output_pre)/((np.exp(FCf_output_pre) + 1)**2)
	FCi_output_rev = np.exp(FCi_output_pre)/((np.exp(FCi_output_pre) + 1)**2)
	FCo_output_rev = np.exp(FCo_output_pre)/((np.exp(FCo_output_pre) + 1)**2)
	FCm_output_rev = 1
	
	
	# choose action
	CHANCE_RAND = np.max((1 - ((1-EPS_GREED_FINAL)/EPS_GREED_FINAL_TIME)*step, EPS_GREED_FINAL))
	if np.random.rand() <= CHANCE_RAND:
		action = np.random.randint(4)
	else:
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
			r += reward_phase
			if reward_phase == 1:
				reward_phase = -1
			else:
				reward_phase = 1
			
			red_axis = np.random.randint(2)
			red_direction = np.random.randint(2)

			if red_direction == 1:
				reds[red_axis] = 32
			else:
				reds[red_axis] = -SCALE

			reds[1 - red_axis] = (32+SCALE) * np.random.random() - SCALE
			show_transition = 255

	# move blocks
	reds[red_axis] -= RED_MOV_RATE * (2*red_direction - 1)
	
	# have any blocks moved off screen?
	if reds[red_axis] < -SCALE or reds[red_axis] > 32:
		r += reward_phase/4.
		if reward_phase == 1:
			reward_phase = -1
		else:
			reward_phase = 1
		
		red_axis = np.random.randint(2)
		red_direction = np.random.randint(2)

		if red_direction == 1:
			reds[red_axis] = 32
		else:
			reds[red_axis] = -SCALE

		reds[1 - red_axis] = (32+SCALE) * np.random.random() - SCALE
		show_transition = 255
	
	r_total += r
	
	# debug/visualization
	save_loc = step % SAVE_FREQ
	
	CEC_recent[save_loc] = copy.deepcopy(CEC)
	action_recent[save_loc] = action
	r_recent[save_loc] = r
	imgs_recent[save_loc] = copy.deepcopy(img[0])
	
	# show blocks
	img = np.zeros((1,3,32,32),dtype='single')
	img[0,2,np.max((np.round(reds[0]),0)):np.round(reds[0]+SCALE), np.max((np.round(reds[1]),0)):np.round(reds[1]+SCALE)] = 255
	img[0,1,player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] = 255
	img[0,reward_phase+1,player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] = show_transition

	show_transition = 0
	
	# forward pass prev network
	set_buffer(img - img_mean, IMGS_PAD, gpu=GPU_PREV)
	
	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_PREV)
	conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_PREV)
	conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_PREV)

	# compute target
	max_output3_prev = return_buffer(MAX_OUTPUT3, gpu=GPU_PREV)
	
	FCm_output_pre = np.einsum(FCm_prev, range(4), max_output3_prev, [4, 1,2,3], [4, 0]) + Bm_prev
	FCi_output_pre = np.einsum(FCi_prev, range(4), max_output3_prev, [4, 1,2,3], [4, 0]) + Bi_prev
	FCo_output_pre = np.einsum(FCo_prev, range(4), max_output3_prev, [4, 1,2,3], [4, 0]) + Bo_prev
	FCf_output_pre = np.einsum(FCf_prev, range(4), max_output3_prev, [4, 1,2,3], [4, 0]) + Bf_prev
	
	FCf_output_prev = 1 / (1 + np.exp(-FCf_output_pre))
	FCi_output_prev = 1 / (1 + np.exp(-FCi_output_pre))
	FCo_output_prev = 1 / (1 + np.exp(-FCo_output_pre))
	FCm_output_prev = FCm_output_pre
	
	FC_output_prev = FCo_output_prev * (FCf_output_prev * CEC_prev + FCi_output_prev * FCm_output_prev)
	
	CEC_prev = CEC_prev*FCf_output_prev + FCi_output_prev*FCm_output_prev
	
	FC2f_output_pre = np.einsum(FC2f_prev, [0,1], FC_output_prev, [2,1], [2,0]) + B2f_prev
	FC2o_output_pre = np.einsum(FC2o_prev, [0,1], FC_output_prev, [2,1], [2,0]) + B2o_prev
	FC2i_output_pre = np.einsum(FC2i_prev, [0,1], FC_output_prev, [2,1], [2,0]) + B2i_prev
	FC2m_output_pre = np.einsum(FC2m_prev, [0,1], FC_output_prev, [2,1], [2,0]) + B2m_prev
	
	FC2f_output_prev = 1 / (1 + np.exp(-FC2f_output_pre))
	FC2o_output_prev = 1 / (1 + np.exp(-FC2o_output_pre))
	FC2i_output_prev = 1 / (1 + np.exp(-FC2i_output_pre))
	FC2m_output_prev = FC2m_output_pre
	
	FC2_output_prev = FC2o_output_prev * (FC2f_output_prev * CEC2_prev + FC2i_output_prev * FC2m_output_prev)
	
	CEC2_prev = CEC2_prev*FC2f_output_prev + FC2i_output_prev*FC2m_output_prev
	
	FC3f_output_pre = np.einsum(FC3f_prev, [0,1], FC2_output, [2,1], [2,0]) + B3f_prev
	FC3o_output_pre = np.einsum(FC3o_prev, [0,1], FC2_output, [2,1], [2,0]) + B3o_prev
	FC3i_output_pre = np.einsum(FC3i_prev, [0,1], FC2_output, [2,1], [2,0]) + B3i_prev
	FC3m_output_pre = np.einsum(FC3m_prev, [0,1], FC2_output, [2,1], [2,0]) + B3m_prev
	
	FC3f_output_prev = 1 / (1 + np.exp(-FC3f_output_pre))
	FC3o_output_prev = 1 / (1 + np.exp(-FC3o_output_pre))
	FC3i_output_prev = 1 / (1 + np.exp(-FC3i_output_pre))
	FC3m_output_prev = FC3m_output_pre
	
	FC3_output_prev = FC3o_output_prev * (FC3f_output_prev * CEC3_prev + FC3i_output_prev * FC3m_output_prev)
	
	CEC3_prev = CEC3_prev*FC3f_output_prev + FC3i_output_prev*FC3m_output_prev
	
	pred_prev = np.einsum(FL_prev, [0,1], FC3_output_prev, [2, 1], [0])
	
	y_output = r + GAMMA * np.max(pred_prev)
		
	############## backprop
	
	pred_m_Y = y_output - pred[action]
	
	err += pred_m_Y**2
	

	############ FL
	
	#dFL = np.einsum(pred_m_Y, [0,1], FC2_output, [0,2], [1,2])
	
	above_w = np.dot(pred_m_Y, FL[action])
	
	######################### mem 3 gradients:
	
	FC3f_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3f * FC3f_output + FC3f_output_rev * CEC3)
	FC3i_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3i * FC3f_output + FC3i_output_rev * FC3m_output)
	FC3m_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3m * FC3f_output + FC3i_output * FC3m_output_rev)
	FC3o_output_rev_sig = above_w * FC3o_output_rev * (FC3f_output * CEC3 + FC3i_output * FC3m_output)
	
	CEC3_dFC3f = CEC3_dFC3f * FC3f_output + FC3f_output_rev * CEC3
	CEC3_dFC3m = CEC3_dFC3m * FC3f_output + FC3i_output * FC3m_output_rev
	CEC3_dFC3i = CEC3_dFC3i * FC3f_output + FC3i_output_rev * FC3m_output
	
	dB3f += np.squeeze(FC3f_output_rev_sig)
	dB3i += np.squeeze(FC3i_output_rev_sig)
	dB3m += np.squeeze(FC3m_output_rev_sig)
	dB3o += np.squeeze(FC3o_output_rev_sig)
	
	dFC3f += np.einsum(FC2_output, [0,1], FC3f_output_rev_sig, [0,2], [2,1])
	dFC3i += np.einsum(FC2_output, [0,1], FC3i_output_rev_sig, [0,2], [2,1])
	dFC3m += np.einsum(FC2_output, [0,1], FC3m_output_rev_sig, [0,2], [2,1])
	dFC3o += np.einsum(FC2_output, [0,1], FC3o_output_rev_sig, [0,2], [2,1])
	
	above_w = np.einsum(FC3o, [0,1], FC3o_output_rev_sig, [2,0], [2,1])
	above_w += np.einsum(FC3f, [0,1], FC3f_output_rev_sig, [2,0], [2,1])
	above_w += np.einsum(FC3i, [0,1], FC3i_output_rev_sig, [2,0], [2,1])
	above_w += np.einsum(FC3m, [0,1], FC3m_output_rev_sig, [2,0], [2,1])
	
	######################### mem 2 gradients:
	
	FC2f_output_rev_sig = above_w * FC2o_output * (FC2f_output_rev * CEC2)
	FC2i_output_rev_sig = above_w * FC2o_output * (FC2i_output_rev * FC2m_output)
	FC2m_output_rev_sig = above_w * FC2o_output * (FC2i_output * FC2m_output_rev)
	FC2o_output_rev_sig = above_w * FC2o_output_rev * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	CEC2_dFC2f = CEC2_dFC2f * FC2f_output + FC2f_output_rev * CEC2
	CEC2_dFC2m = CEC2_dFC2m * FC2f_output + FC2i_output * FC2m_output_rev
	CEC2_dFC2i = CEC2_dFC2i * FC2f_output + FC2i_output_rev * FC2m_output
	
	dB2f += np.squeeze(FC2f_output_rev_sig)
	dB2i += np.squeeze(FC2i_output_rev_sig)
	dB2m += np.squeeze(FC2m_output_rev_sig)
	dB2o += np.squeeze(FC2o_output_rev_sig)
	
	dFC2f += np.einsum(FC_output, [0,1], FC2f_output_rev_sig, [0,2], [2,1])
	dFC2i += np.einsum(FC_output, [0,1], FC2i_output_rev_sig, [0,2], [2,1])
	dFC2m += np.einsum(FC_output, [0,1], FC2m_output_rev_sig, [0,2], [2,1])
	dFC2o += np.einsum(FC_output, [0,1], FC2o_output_rev_sig, [0,2], [2,1])
	
	above_w = np.einsum(FC2o, [0,1], FC2o_output_rev_sig, [2,0], [2,1])
	above_w += np.einsum(FC2f, [0,1], FC2f_output_rev_sig, [2,0], [2,1])
	above_w += np.einsum(FC2i, [0,1], FC2i_output_rev_sig, [2,0], [2,1])
	above_w += np.einsum(FC2m, [0,1], FC2m_output_rev_sig, [2,0], [2,1])
	
	########################## mem 1 gradients:

	FCf_output_rev_sig = above_w * FCo_output * (FCf_output_rev * CEC)
	FCi_output_rev_sig = above_w * FCo_output * (FCi_output_rev * FCm_output)
	FCm_output_rev_sig = above_w * FCo_output * (FCi_output * FCm_output_rev)
	FCo_output_rev_sig = above_w * FCo_output_rev * (FCf_output * CEC + FCi_output * FCm_output)
	
	CEC_dFCf = CEC_dFCf * FCf_output + FCf_output_rev * CEC
	CEC_dFCm = CEC_dFCm * FCf_output + FCi_output * FCm_output_rev
	CEC_dFCi = CEC_dFCi * FCf_output + FCi_output_rev * FCm_output
	
	dBf += np.squeeze(FCf_output_rev_sig)
	dBi += np.squeeze(FCi_output_rev_sig)
	dBm += np.squeeze(FCm_output_rev_sig)
	dBo += np.squeeze(FCo_output_rev_sig)
	
	dFCf += np.einsum(max_output3, range(4), FCf_output_rev_sig, [0,4], [4,1,2,3])
	dFCi += np.einsum(max_output3, range(4), FCi_output_rev_sig, [0,4], [4,1,2,3])
	dFCm += np.einsum(max_output3, range(4), FCm_output_rev_sig, [0,4], [4,1,2,3])
	dFCo += np.einsum(max_output3, range(4), FCo_output_rev_sig, [0,4], [4,1,2,3])
	
	above_w = np.einsum(FCo, range(4), FCo_output_rev_sig, [4,0], [4,1,2,3])
	above_w += np.einsum(FCi, range(4), FCi_output_rev_sig, [4,0], [4,1,2,3])
	above_w += np.einsum(FCm, range(4), FCm_output_rev_sig, [4,0], [4,1,2,3])
	above_w += np.einsum(FCf, range(4), FCf_output_rev_sig, [4,0], [4,1,2,3])
	
	
	set_buffer(above_w, FL_PRED, gpu=GPU_CUR)
	
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
	dFL[action] += FC3_output[0]*pred_m_Y
	
	dF3 += return_buffer(DF3, stream=3, gpu=GPU_CUR)
	dF2 += return_buffer(DF2, stream=2, gpu=GPU_CUR)
	dF1 += return_buffer(DF1, stream=1, gpu=GPU_CUR)
	
	#### update filter weights
	if step % BATCH_SZ == 0:
		F1 += dF1*EPS / BATCH_SZ
		F2 += dF2*EPS / BATCH_SZ
		F3 += dF3*EPS / BATCH_SZ
		FL += dFL*EPS / BATCH_SZ
		
		FCf += dFCf*EPS / BATCH_SZ
		FCo += dFCo*EPS / BATCH_SZ
		FCm += dFCm*EPS / BATCH_SZ
		FCi += dFCi*EPS / BATCH_SZ
		
		Bf += dBf*EPS / BATCH_SZ
		Bo += dBo*EPS / BATCH_SZ
		Bm += dBm*EPS / BATCH_SZ
		Bi += dBi*EPS / BATCH_SZ
		
		FC2f += dFC2f*EPS / BATCH_SZ
		FC2o += dFC2o*EPS / BATCH_SZ
		FC2m += dFC2m*EPS / BATCH_SZ
		FC2i += dFC2i*EPS / BATCH_SZ
		
		B2f += dB2f*EPS / BATCH_SZ
		B2o += dB2o*EPS / BATCH_SZ
		B2m += dB2m*EPS / BATCH_SZ
		B2i += dB2i*EPS / BATCH_SZ
		
		FC3f += dFC3f*EPS / BATCH_SZ
		FC3o += dFC3o*EPS / BATCH_SZ
		FC3m += dFC3m*EPS / BATCH_SZ
		FC3i += dFC3i*EPS / BATCH_SZ
		
		B3f += dB3f*EPS / BATCH_SZ
		B3o += dB3o*EPS / BATCH_SZ
		B3m += dB3m*EPS / BATCH_SZ
		B3i += dB3i*EPS / BATCH_SZ
		
		set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
		set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
		set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
		
		dF1 = np.zeros_like(F1)
		dF2 = np.zeros_like(F2)
		dF3 = np.zeros_like(F3)
		dFL = np.zeros_like(FL)

		dFCm = np.zeros_like(FCm)
		dFCi = np.zeros_like(FCi)
		dFCo = np.zeros_like(FCo)
		dFCf = np.zeros_like(FCf)

		dBm = np.zeros_like(Bm)
		dBi = np.zeros_like(Bi)
		dBo = np.zeros_like(Bo)
		dBf = np.zeros_like(Bf)

		dFC2m = np.zeros_like(FC2m)
		dFC2i = np.zeros_like(FC2i)
		dFC2o = np.zeros_like(FC2o)
		dFC2f = np.zeros_like(FC2f)

		dB2m = np.zeros_like(B2m)
		dB2i = np.zeros_like(B2i)
		dB2o = np.zeros_like(B2o)
		dB2f = np.zeros_like(B2f)

		dFC3m = np.zeros_like(FC3m)
		dFC3i = np.zeros_like(FC3i)
		dFC3o = np.zeros_like(FC3o)
		dFC3f = np.zeros_like(FC3f)

		dB3m = np.zeros_like(B3m)
		dB3i = np.zeros_like(B3i)
		dB3o = np.zeros_like(B3o)
		dB3f = np.zeros_like(B3f)
		
		network_updates += 1
		
		if network_updates % NETWORK_UPDATE == 0:
			print 'updating network'
			FCm_prev = copy.deepcopy(FCm)
			FCi_prev = copy.deepcopy(FCi)
			FCo_prev = copy.deepcopy(FCo)
			FCf_prev = copy.deepcopy(FCf)

			Bm_prev = copy.deepcopy(Bm)
			Bi_prev = copy.deepcopy(Bi)
			Bo_prev = copy.deepcopy(Bo)
			Bf_prev = copy.deepcopy(Bf)

			FC2m_prev = copy.deepcopy(FC2m)
			FC2i_prev = copy.deepcopy(FC2i)
			FC2o_prev = copy.deepcopy(FC2o)
			FC2f_prev = copy.deepcopy(FC2f)

			B2m_prev = copy.deepcopy(B2m)
			B2i_prev = copy.deepcopy(B2i)
			B2o_prev = copy.deepcopy(B2o)
			B2f_prev = copy.deepcopy(B2f)

			FC3m_prev = copy.deepcopy(FC3m)
			FC3i_prev = copy.deepcopy(FC3i)
			FC3o_prev = copy.deepcopy(FC3o)
			FC3f_prev = copy.deepcopy(FC3f)

			B3m_prev = copy.deepcopy(B3m)
			B3i_prev = copy.deepcopy(B3i)
			B3o_prev = copy.deepcopy(B3o)
			B3f_prev = copy.deepcopy(B3f)

			FL_prev = copy.deepcopy(FL)
			
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)
				
	step += 1
	
	if step % SAVE_FREQ == 0:
		r_total_plot.append(r_total)
		err_plot.append(err)
		
		savemat(file_name, {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, 'F1_init': F1_init, 'step': step, 'img': img, 'err_plot': err_plot, 'CEC':CEC, 'imgs_mean_player': imgs_mean_player, 'imgs_mean_red': imgs_mean_red, \
				'CEC_recent':CEC_recent, 'action_recent': action_recent, 'r_recent':r_recent,'SAVE_FREQ':SAVE_FREQ,\
				'imgs_recent': imgs_recent, 'FCi': FCi, 'FCm': FCm, 'FCf': FCf, 'FCo': FCo, 'EPS':EPS, 'NETWORK_UPDATE':NETWORK_UPDATE,\
				'FC2i': FCi, 'FC2m': FC2m, 'FC2f': FC2f, 'FC2o': FC2o,\
				'FC3i': FCi, 'FC3m': FC3m, 'FC3f': FC3f, 'FC3o': FC3o,'MEM_SZ':MEM_SZ,'EPS_GREED_FINAL_TIME':EPS_GREED_FINAL_TIME})
		
		conv_output1 = return_buffer(CONV_OUTPUT1, gpu=GPU_CUR)
		conv_output2 = return_buffer(CONV_OUTPUT2, gpu=GPU_CUR)
		conv_output3 = return_buffer(CONV_OUTPUT3, gpu=GPU_CUR)
		print '---------------------------------------------'
		print step, 'err:',err, 'r:',r_total, 'updates:',network_updates, 'eps:', CHANCE_RAND, 't:',time.time() - t_start, file_name
		ft = EPS/BATCH_SZ
		print 'F1: %f %f (%f);   layer: %f %f' % (np.min(F1), np.max(F1), np.median(np.abs(ft*dF1.ravel())/np.abs(F1.ravel())),\
						np.min(conv_output1), np.max(conv_output1))
		print 'F2: %f %f (%f)   layer: %f %f' % (np.min(F2), np.max(F2), np.median(np.abs(ft*dF2.ravel())/np.abs(F2.ravel())),\
						np.min(conv_output2), np.max(conv_output2))
		print 'F3: %f %f (%f)   layer: %f %f' % (np.min(F3), np.max(F3), np.median(np.abs(ft*dF3.ravel())/np.abs(F3.ravel())),\
						np.min(conv_output3), np.max(conv_output3))
		print
		print 'FCm: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCm), np.max(FCm), np.median(np.abs(ft*dFCm.ravel())/np.abs(FCm.ravel())),\
						np.min(FCm_output), np.max(FCm_output), np.median(FCm_output))
		print 'FCf: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCf), np.max(FCf), np.median(np.abs(ft*dFCf.ravel())/np.abs(FCf.ravel())),\
						np.min(FCf_output), np.max(FCf_output), np.median(FCf_output))
		print 'FCi: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCi), np.max(FCi), np.median(np.abs(ft*dFCi.ravel())/np.abs(FCi.ravel())),\
						np.min(FCi_output), np.max(FCi_output), np.median(FCi_output))
		print 'FCo: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCo), np.max(FCo), np.median(np.abs(ft*dFCo.ravel())/np.abs(FCo.ravel())),\
						np.min(FCo_output), np.max(FCo_output), np.median(FCo_output))
		print 'CEC: %f %f, kept: %f %f, new: %f %f' % (np.min(CEC), np.max(CEC), np.min(CEC_kept), np.max(CEC_kept), \
						np.min(CEC_new), np.max(CEC_new))
		print 'FC.: %f %f' % (np.min(FC_output), np.max(FC_output))
		print
		print 'FC2m: %f %f (%f)   layer: %f %f (%f)' % (np.min(FC2m), np.max(FC2m), np.median(np.abs(ft*dFC2m.ravel())/np.abs(FC2m.ravel())),\
						np.min(FC2m_output), np.max(FC2m_output), np.median(FC2m_output))
		print 'FC2f: %f %f (%f)   layer: %f %f (%f)' % (np.min(FC2f), np.max(FC2f), np.median(np.abs(ft*dFC2f.ravel())/np.abs(FC2f.ravel())),\
						np.min(FC2f_output), np.max(FC2f_output), np.median(FC2f_output))
		print 'FC2i: %f %f (%f)   layer: %f %f (%f)' % (np.min(FC2i), np.max(FC2i), np.median(np.abs(ft*dFC2i.ravel())/np.abs(FC2i.ravel())),\
						np.min(FC2i_output), np.max(FC2i_output), np.median(FC2i_output))
		print 'FC2o: %f %f (%f)   layer: %f %f (%f)' % (np.min(FC2o), np.max(FC2o), np.median(np.abs(ft*dFC2o.ravel())/np.abs(FC2o.ravel())),\
						np.min(FC2o_output), np.max(FC2o_output), np.median(FC2o_output))
		print 'CEC2: %f %f, kept: %f %f, new: %f %f' % (np.min(CEC2), np.max(CEC2), np.min(CEC2_kept), np.max(CEC2_kept), \
						np.min(CEC2_new), np.max(CEC2_new))
		print 'FC2.: %f %f' % (np.min(FC2_output), np.max(FC2_output))
		print
		print 'FC3m: %f %f (%f)   layer: %f %f (%f)' % (np.min(FC3m), np.max(FC3m), np.median(np.abs(ft*dFC3m.ravel())/np.abs(FC3m.ravel())),\
						np.min(FC3m_output), np.max(FC3m_output), np.median(FC3m_output))
		print 'FC3f: %f %f (%f)   layer: %f %f (%f)' % (np.min(FC3f), np.max(FC3f), np.median(np.abs(ft*dFC3f.ravel())/np.abs(FC3f.ravel())),\
						np.min(FC3f_output), np.max(FC3f_output), np.median(FC3f_output))
		print 'FC3i: %f %f (%f)   layer: %f %f (%f)' % (np.min(FC3i), np.max(FC3i), np.median(np.abs(ft*dFC3i.ravel())/np.abs(FC3i.ravel())),\
						np.min(FC3i_output), np.max(FC3i_output), np.median(FC3i_output))
		print 'FC3o: %f %f (%f)   layer: %f %f (%f)' % (np.min(FC3o), np.max(FC3o), np.median(np.abs(ft*dFC3o.ravel())/np.abs(FC3o.ravel())),\
						np.min(FC3o_output), np.max(FC3o_output), np.median(FC3o_output))
		print 'CEC3: %f %f, kept: %f %f, new: %f %f' % (np.min(CEC3), np.max(CEC3), np.min(CEC3_kept), np.max(CEC3_kept), \
						np.min(CEC3_new), np.max(CEC3_new))
		print 'FC3.: %f %f' % (np.min(FC3_output), np.max(FC3_output))
		print
		print 'FL: %f %f (%f)   layer: %f %f' % (np.min(FL), np.max(FL), np.median(np.abs(ft*dFL.ravel())/np.abs(FL.ravel())),\
						np.min(pred), np.max(pred))

		
		err = 0
		r_total = 0
		
		t_start = time.time()
