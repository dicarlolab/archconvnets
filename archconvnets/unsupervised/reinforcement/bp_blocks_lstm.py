from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random

#kernprof -l bp_cudnn.py
#python -m line_profiler bp_cudnn.py.lprof  > p
#@profile
#def sf():
file_name = '/home/darren/reinforcement_blocks_lstm_slower.mat'

EPS_GREED_FINAL = .1
EPS_GREED_FINAL_TIME = 5000000/10
GAMMA = 0.99

NETWORK_UPDATE = 10000
EPS = 1e-2
EPS_F = 1e-3
SAVE_FREQ = 1000

SCALE = 6
MAX_LOC = 32 - SCALE
N_MEAN = 500#000

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.001
F3_scale = 0.001
FL_scale = .03

N = 32
n1 = N # L1 filters
n2 = N # ...
n3 = N

n1m = 128
n2m = 128+1
n3m = 128+2

s3 = 3 # L1 filter size (px)
s2 = 4 # ...
s1 = 5

N_C = 4 # directions L, R, U, D

PLAYER_MOV_RATE = 3
RED_MOV_RATE = 2
BLUE_MOV_RATE = 2

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


np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))

n_in = n3*max_output_sz3**2

## l1
FCm = 2*np.single(np.random.random(size=(n1m, n_in))) - 1
FCi = 2*np.single(np.random.random(size=(n1m, n_in))) - 1
FCo = 2*np.single(np.random.random(size=(n1m, n_in))) - 1
FCf = 2*np.single(np.random.random(size=(n1m, n_in))) - 1

Bm = np.zeros_like(2*np.single(np.random.random(size=n1m)) - 1) ##
Bi = np.zeros_like(2*np.single(np.random.random(size=n1m)) - 1)
Bo = 2*np.ones_like(2*np.single(np.random.random(size=n1m)) - 1)
Boo = np.zeros_like(2*np.single(np.random.random(size=n1m)) - 1) ##
Bf = -2*np.ones_like(2*np.single(np.random.random(size=n1m)) - 1)
CEC = np.zeros_like(2*np.single(np.random.random(size=n1m)) - 1)

CEC_dFCm = np.zeros_like(CEC)
CEC_dFCi = np.zeros_like(CEC)
CEC_dFCf = np.zeros_like(CEC)

## l2
FCm2 = 2*np.single(np.random.random(size=(n2m, n1m))) - 1
FCi2 = 2*np.single(np.random.random(size=(n2m, n1m))) - 1
FCo2 = 2*np.single(np.random.random(size=(n2m, n1m))) - 1
FCf2 = 2*np.single(np.random.random(size=(n2m, n1m))) - 1

Bm2 = np.zeros_like(2*np.single(np.random.random(size=n2m)) - 1) ##
Bi2 = np.zeros_like(2*np.single(np.random.random(size=n2m)) - 1)
Bo2 = 2*np.ones_like(2*np.single(np.random.random(size=n2m)) - 1)
Boo2 = np.zeros_like(2*np.single(np.random.random(size=n2m)) - 1) ##
Bf2 = -2*np.ones_like(2*np.single(np.random.random(size=n2m)) - 1)
CEC2 = np.zeros_like(2*np.single(np.random.random(size=n2m)) - 1)

CEC2_dFCm2 = np.zeros_like(CEC2)
CEC2_dFCi2 = np.zeros_like(CEC2)
CEC2_dFCf2 = np.zeros_like(CEC2)

## l3
FCm3 = 2*np.single(np.random.random(size=(n3m, n2m))) - 1
FCi3 = 2*np.single(np.random.random(size=(n3m, n2m))) - 1
FCo3 = 2*np.single(np.random.random(size=(n3m, n2m))) - 1
FCf3 = 2*np.single(np.random.random(size=(n3m, n2m))) - 1

Bm3 = np.zeros_like(2*np.single(np.random.random(size=n3m)) - 1) ##
Bi3 = np.zeros_like(2*np.single(np.random.random(size=n3m)) - 1)
Bo3 = 2*np.ones_like(2*np.single(np.random.random(size=n3m)) - 1)
Boo3 = np.zeros_like(2*np.single(np.random.random(size=n3m)) - 1) ##
Bf3 = -2*np.ones_like(2*np.single(np.random.random(size=n3m)) - 1)
CEC3 = np.zeros_like(2*np.single(np.random.random(size=n3m)) - 1)

CEC3_dFCm3 = np.zeros_like(CEC3)
CEC3_dFCi3 = np.zeros_like(CEC3)
CEC3_dFCf3 = np.zeros_like(CEC3)

FL = np.single(np.random.normal(scale=1, size=(4,n3m)))

CEC_kept = 0; CEC_new = 0
CEC_kept2 = 0; CEC_new2 = 0
CEC_kept3 = 0; CEC_new3 = 0

###
CEC_prev_prev = copy.deepcopy(CEC)
CEC2_prev_prev = copy.deepcopy(CEC2)
CEC3_prev_prev = copy.deepcopy(CEC3)

FCm_prev = copy.deepcopy(FCm)
FCi_prev = copy.deepcopy(FCi)
FCo_prev = copy.deepcopy(FCo)
FCf_prev = copy.deepcopy(FCf)

Boo_prev = copy.deepcopy(Boo)
Bm_prev = copy.deepcopy(Bm)
Bi_prev = copy.deepcopy(Bi)
Bo_prev = copy.deepcopy(Bo)
Bf_prev = copy.deepcopy(Bf)

FCm2_prev = copy.deepcopy(FCm2)
FCi2_prev = copy.deepcopy(FCi2)
FCo2_prev = copy.deepcopy(FCo2)
FCf2_prev = copy.deepcopy(FCf2)

Boo2_prev = copy.deepcopy(Boo2)
Bm2_prev = copy.deepcopy(Bm2)
Bi2_prev = copy.deepcopy(Bi2)
Bo2_prev = copy.deepcopy(Bo2)
Bf2_prev = copy.deepcopy(Bf2)

FCm3_prev = copy.deepcopy(FCm3)
FCi3_prev = copy.deepcopy(FCi3)
FCo3_prev = copy.deepcopy(FCo3)
FCf3_prev = copy.deepcopy(FCf3)

Boo3_prev = copy.deepcopy(Boo3)
Bm3_prev = copy.deepcopy(Bm3)
Bi3_prev = copy.deepcopy(Bi3)
Bo3_prev = copy.deepcopy(Bo3)
Bf3_prev = copy.deepcopy(Bf3)

FL_prev = copy.deepcopy(FL)

set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)

set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)

F1_init = copy.deepcopy(F1)

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
action_recent = np.zeros(SAVE_FREQ, dtype='int')
r_recent = np.zeros(SAVE_FREQ, dtype='single')

imgs_mean_red = np.zeros((32,32))
imgs_mean_player = np.zeros((32,32))

t_start = time.time()

######## mean img
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
	
	inputs = max_output3.ravel()
	
	## forward pass
	
	## mem 1
	FCi_output_pre = np.dot(FCi, inputs) + Bi
	FCf_output_pre = np.dot(FCf, inputs) + Bf
	FCm_output_pre = np.dot(FCm, inputs) + Bm
	
	FCf_output = 1 / (1 + np.exp(-FCf_output_pre))
	FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
	FCm_output = 1 / (1 + np.exp(-FCm_output_pre)) - .5
	
	CEC_kept = CEC*FCf_output
	CEC_new = FCi_output*FCm_output
	
	CEC_prev = copy.deepcopy(CEC)
	CEC = CEC*FCf_output + FCi_output*FCm_output
	
	FCo_output_pre = np.dot(FCo, inputs) + Bo
	FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
	
	FC_output = FCo_output * CEC + Boo
	
	## mem 2
	FCi2_output_pre = np.dot(FCi2, FC_output) + Bi2
	FCf2_output_pre = np.dot(FCf2, FC_output) + Bf2
	FCm2_output_pre = np.dot(FCm2, FC_output) + Bm2
	
	FCf2_output = 1 / (1 + np.exp(-FCf2_output_pre))
	FCi2_output = 1 / (1 + np.exp(-FCi2_output_pre))
	FCm2_output = 1 / (1 + np.exp(-FCm2_output_pre)) - .5
	
	CEC_kept2 = CEC2*FCf2_output
	CEC_new2 = FCi2_output*FCm2_output
	
	CEC2_prev = copy.deepcopy(CEC2)
	CEC2 = CEC2*FCf2_output + FCi2_output*FCm2_output
	
	FCo2_output_pre = np.dot(FCo2, FC_output) + Bo2
	FCo2_output = 1 / (1 + np.exp(-FCo2_output_pre))
	
	FC2_output = FCo2_output * (FCf2_output * CEC2 + FCi2_output * FCm2_output)
	
	## mem 3
	FCi3_output_pre = np.dot(FCi3, FC2_output) + Bi3
	FCf3_output_pre = np.dot(FCf3, FC2_output) + Bf3
	FCm3_output_pre = np.dot(FCm3, FC2_output) + Bm3
	
	FCf3_output = 1 / (1 + np.exp(-FCf3_output_pre))
	FCi3_output = 1 / (1 + np.exp(-FCi3_output_pre))
	FCm3_output = 1 / (1 + np.exp(-FCm3_output_pre)) - .5
	
	CEC_kept3 = CEC3*FCf3_output
	CEC_new3 = FCi3_output*FCm3_output
	
	CEC3_prev = copy.deepcopy(CEC3)
	CEC3 = CEC3*FCf3_output + FCi3_output*FCm3_output
	
	FCo3_output_pre = np.dot(FCo3, FC2_output) + Bo3
	FCo3_output = 1 / (1 + np.exp(-FCo3_output_pre))
	
	FC3_output = FCo3_output * (FCf3_output * CEC3 + FCi3_output * FCm3_output)
	
	pred = np.dot(FL, FC3_output)
	
	############### reverse pointwise
	FCf_output_rev = np.exp(FCf_output_pre)/((np.exp(FCf_output_pre) + 1)**2)
	FCi_output_rev = np.exp(FCi_output_pre)/((np.exp(FCi_output_pre) + 1)**2)
	FCo_output_rev = np.exp(FCo_output_pre)/((np.exp(FCo_output_pre) + 1)**2)
	FCm_output_rev = np.exp(FCm_output_pre)/((np.exp(FCm_output_pre) + 1)**2)
	
	FCf2_output_rev = np.exp(FCf2_output_pre)/((np.exp(FCf2_output_pre) + 1)**2)
	FCi2_output_rev = np.exp(FCi2_output_pre)/((np.exp(FCi2_output_pre) + 1)**2)
	FCo2_output_rev = np.exp(FCo2_output_pre)/((np.exp(FCo2_output_pre) + 1)**2)
	FCm2_output_rev = np.exp(FCm2_output_pre)/((np.exp(FCm2_output_pre) + 1)**2)
	
	FCf3_output_rev = np.exp(FCf3_output_pre)/((np.exp(FCf3_output_pre) + 1)**2)
	FCi3_output_rev = np.exp(FCi3_output_pre)/((np.exp(FCi3_output_pre) + 1)**2)
	FCo3_output_rev = np.exp(FCo3_output_pre)/((np.exp(FCo3_output_pre) + 1)**2)
	FCm3_output_rev = np.exp(FCm3_output_pre)/((np.exp(FCm3_output_pre) + 1)**2)
	
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
	
	action_recent[save_loc] = action
	r_recent[save_loc] = r
	imgs_recent[save_loc] = copy.deepcopy(img[0])
	
	# show blocks
	img = np.zeros((1,3,32,32),dtype='single')
	img[0,2,np.max((np.round(reds[0]),0)):np.round(reds[0]+SCALE), np.max((np.round(reds[1]),0)):np.round(reds[1]+SCALE)] = 255
	img[0,1,player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] = 255
	img[0,reward_phase+1,player[0]:(player[0]+SCALE), player[1]:player[1]+SCALE] = show_transition

	show_transition = 0
	
	##################
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
	
	inputs_prev = max_output3_prev.ravel()
	
	## mem 1
	FCi_output_pre = np.dot(FCi_prev, inputs_prev) + Bi_prev
	FCf_output_pre = np.dot(FCf_prev, inputs_prev) + Bf_prev
	FCm_output_pre = np.dot(FCm_prev, inputs_prev) + Bm_prev
	
	FCf_output_prev = 1 / (1 + np.exp(-FCf_output_pre))
	FCi_output_prev = 1 / (1 + np.exp(-FCi_output_pre))
	FCm_output_prev = 1 / (1 + np.exp(-FCm_output_pre)) - .5
	
	CEC_prev_prev = CEC_prev_prev*FCf_output_prev + FCi_output_prev*FCm_output_prev
	
	FCo_output_pre = np.dot(FCo_prev, inputs_prev) + Bo_prev
	FCo_output_prev = 1 / (1 + np.exp(-FCo_output_pre))
	
	FC_output_prev = FCo_output_prev * CEC_prev_prev + Boo_prev
	
	## mem 2
	FCi2_output_pre = np.dot(FCi2_prev, FC_output_prev) + Bi2_prev
	FCf2_output_pre = np.dot(FCf2_prev, FC_output_prev) + Bf2_prev
	FCm2_output_pre = np.dot(FCm2_prev, FC_output_prev) + Bm2_prev
	
	FCf2_output_prev = 1 / (1 + np.exp(-FCf2_output_pre))
	FCi2_output_prev = 1 / (1 + np.exp(-FCi2_output_pre))
	FCm2_output_prev = 1 / (1 + np.exp(-FCm2_output_pre)) - .5
	
	CEC2_prev_prev = CEC2_prev_prev*FCf2_output_prev + FCi2_output_prev*FCm2_output_prev
	
	FCo2_output_pre = np.dot(FCo2_prev, FC_output_prev) + Bo2_prev
	FCo2_output_prev = 1 / (1 + np.exp(-FCo2_output_pre))
	
	FC2_output_prev = FCo2_output_prev * (FCf2_output_prev * CEC2_prev_prev + FCi2_output_prev * FCm2_output_prev)
	
	## mem 3
	FCi3_output_pre = np.dot(FCi3_prev, FC2_output_prev) + Bi3_prev
	FCf3_output_pre = np.dot(FCf3_prev, FC2_output_prev) + Bf3_prev
	FCm3_output_pre = np.dot(FCm3_prev, FC2_output_prev) + Bm3_prev
	
	FCf3_output_prev = 1 / (1 + np.exp(-FCf3_output_pre))
	FCi3_output_prev = 1 / (1 + np.exp(-FCi3_output_pre))
	FCm3_output_prev = 1 / (1 + np.exp(-FCm3_output_pre)) - .5
	
	CEC3_prev_prev = CEC3_prev_prev*FCf3_output_prev + FCi3_output_prev*FCm3_output_prev
	
	FCo3_output_pre = np.dot(FCo3_prev, FC2_output_prev) + Bo3_prev
	FCo3_output_prev = 1 / (1 + np.exp(-FCo3_output_pre))
	
	FC3_output_prev = FCo3_output_prev * (FCf3_output_prev * CEC3_prev_prev + FCi3_output_prev * FCm3_output_prev)
	
	pred_prev = np.dot(FL_prev, FC3_output_prev)
	
	y_output = np.single(r + GAMMA * np.max(pred_prev))
	
	############## backprop
	
	pred_m_Y = y_output - pred[action]
	
	err += pred_m_Y**2
	
	above_w = np.dot(pred_m_Y, FL[action])
	
	########################## mem 3 gradients:
	
	dBoo3 = copy.deepcopy(above_w)
	
	FCf3_output_rev_sig = above_w * FCo3_output * (CEC3_dFCf3 * FCf3_output + FCf3_output_rev * CEC3_prev)
	FCi3_output_rev_sig = above_w * FCo3_output * (CEC3_dFCi3 * FCf3_output + FCi3_output_rev * FCm3_output)
	FCm3_output_rev_sig = above_w * FCo3_output * (CEC3_dFCm3 * FCf3_output + FCi3_output * FCm3_output_rev)
	FCo3_output_rev_sig = above_w * FCo3_output_rev * (FCf3_output * CEC3 + FCi3_output * FCm3_output)
	
	CEC3_dFCf3 = CEC3_dFCf3 * FCf3_output + FCf3_output_rev * CEC3_prev
	CEC3_dFCm3 = CEC3_dFCm3 * FCf3_output + FCi3_output * FCm3_output_rev
	CEC3_dFCi3 = CEC3_dFCi3 * FCf3_output + FCi3_output_rev * FCm3_output
	
	dBf3 = FCf3_output_rev_sig
	dBi3 = FCi3_output_rev_sig
	dBm3 = FCm3_output_rev_sig
	dBo3 = FCo3_output_rev_sig
	
	dFCf3 = np.einsum(FC2_output, [0], FCf3_output_rev_sig, [1], [1,0])
	dFCi3 = np.einsum(FC2_output, [0], FCi3_output_rev_sig, [1], [1,0])
	dFCm3 = np.einsum(FC2_output, [0], FCm3_output_rev_sig, [1], [1,0])
	dFCo3 = np.einsum(FC2_output, [0], FCo3_output_rev_sig, [1], [1,0])
	
	above_w = np.einsum(FCo3, [0,1], FCo3_output_rev_sig, [0], [1])
	above_w += np.einsum(FCf3, [0,1], FCf3_output_rev_sig, [0], [1])
	above_w += np.einsum(FCi3, [0,1], FCi3_output_rev_sig, [0], [1])
	above_w += np.einsum(FCm3, [0,1], FCm3_output_rev_sig, [0], [1])
	
	########################## mem 2 gradients:
	
	dBoo2 = copy.deepcopy(above_w)
	
	FCf2_output_rev_sig = above_w * FCo2_output * (CEC2_dFCf2 * FCf2_output + FCf2_output_rev * CEC2_prev)
	FCi2_output_rev_sig = above_w * FCo2_output * (CEC2_dFCi2 * FCf2_output + FCi2_output_rev * FCm2_output)
	FCm2_output_rev_sig = above_w * FCo2_output * (CEC2_dFCm2 * FCf2_output + FCi2_output * FCm2_output_rev)
	FCo2_output_rev_sig = above_w * FCo2_output_rev * (FCf2_output * CEC2 + FCi2_output * FCm2_output)
	
	CEC2_dFCf2 = CEC2_dFCf2 * FCf2_output + FCf2_output_rev * CEC2_prev
	CEC2_dFCm2 = CEC2_dFCm2 * FCf2_output + FCi2_output * FCm2_output_rev
	CEC2_dFCi2 = CEC2_dFCi2 * FCf2_output + FCi2_output_rev * FCm2_output
	
	dBf2 = FCf2_output_rev_sig
	dBi2 = FCi2_output_rev_sig
	dBm2 = FCm2_output_rev_sig
	dBo2 = FCo2_output_rev_sig
	
	dFCf2 = np.einsum(FC_output, [0], FCf2_output_rev_sig, [1], [1,0])
	dFCi2 = np.einsum(FC_output, [0], FCi2_output_rev_sig, [1], [1,0])
	dFCm2 = np.einsum(FC_output, [0], FCm2_output_rev_sig, [1], [1,0])
	dFCo2 = np.einsum(FC_output, [0], FCo2_output_rev_sig, [1], [1,0])
	
	above_w = np.einsum(FCo2, [0,1], FCo2_output_rev_sig, [0], [1])
	above_w += np.einsum(FCf2, [0,1], FCf2_output_rev_sig, [0], [1])
	above_w += np.einsum(FCi2, [0,1], FCi2_output_rev_sig, [0], [1])
	above_w += np.einsum(FCm2, [0,1], FCm2_output_rev_sig, [0], [1])
	
	########################## mem 1 gradients:
	
	dBoo = copy.deepcopy(above_w)
	
	FCf_output_rev_sig = above_w * FCo_output * (CEC_dFCf * FCf_output + FCf_output_rev * CEC_prev)
	FCi_output_rev_sig = above_w * FCo_output * (CEC_dFCi * FCf_output + FCi_output_rev * FCm_output)
	FCm_output_rev_sig = above_w * FCo_output * (CEC_dFCm * FCf_output + FCi_output * FCm_output_rev)
	FCo_output_rev_sig = above_w * FCo_output_rev * (FCf_output * CEC + FCi_output * FCm_output)
	
	CEC_dFCf = CEC_dFCf * FCf_output + FCf_output_rev * CEC_prev
	CEC_dFCm = CEC_dFCm * FCf_output + FCi_output * FCm_output_rev
	CEC_dFCi = CEC_dFCi * FCf_output + FCi_output_rev * FCm_output
	
	dBf = FCf_output_rev_sig
	dBi = FCi_output_rev_sig
	dBm = FCm_output_rev_sig
	dBo = FCo_output_rev_sig
	
	dFCf = np.einsum(inputs, [0], FCf_output_rev_sig, [1], [1,0])
	dFCi = np.einsum(inputs, [0], FCi_output_rev_sig, [1], [1,0])
	dFCm = np.einsum(inputs, [0], FCm_output_rev_sig, [1], [1,0])
	dFCo = np.einsum(inputs, [0], FCo_output_rev_sig, [1], [1,0])
	
	above_w = np.einsum(FCo, [0,1], FCo_output_rev_sig, [0], [1])
	above_w += np.einsum(FCf, [0,1], FCf_output_rev_sig, [0], [1])
	above_w += np.einsum(FCi, [0,1], FCi_output_rev_sig, [0], [1])
	above_w += np.einsum(FCm, [0,1], FCm_output_rev_sig, [0], [1])
	
	above_w = above_w.reshape(max_output3.shape)
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
	dFL = FC3_output[0]*pred_m_Y # for 'action' (FL[action])
	
	dF3 = return_buffer(DF3, stream=3, gpu=GPU_CUR)
	dF2 = return_buffer(DF2, stream=2, gpu=GPU_CUR)
	dF1 = return_buffer(DF1, stream=1, gpu=GPU_CUR)
	
	## update
	F1 += dF1*EPS_F
	F2 += dF2*EPS_F
	F3 += dF3*EPS_F
	FL[action] += dFL*EPS
	
	FCf += dFCf*EPS
	FCo += dFCo*EPS
	FCm += dFCm*EPS
	FCi += dFCi*EPS
	
	Boo += dBoo*EPS
	Bf += dBf*EPS
	Bo += dBo*EPS
	Bm += dBm*EPS
	Bi += dBi*EPS
	
	FCf2 += dFCf2*EPS
	FCo2 += dFCo2*EPS
	FCm2 += dFCm2*EPS
	FCi2 += dFCi2*EPS
	
	Boo2 += dBoo2*EPS
	Bf2 += dBf2*EPS
	Bo2 += dBo2*EPS
	Bm2 += dBm2*EPS
	Bi2 += dBi2*EPS
	
	FCf3 += dFCf3*EPS
	FCo3 += dFCo3*EPS
	FCm3 += dFCm3*EPS
	FCi3 += dFCi3*EPS
	
	Boo3 += dBoo3*EPS
	Bf3 += dBf3*EPS
	Bo3 += dBo3*EPS
	Bm3 += dBm3*EPS
	Bi3 += dBi3*EPS
	
	set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
	set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
	set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
	
	if step % NETWORK_UPDATE == 0:
		print 'updating network'
		FCm_prev = copy.deepcopy(FCm)
		FCi_prev = copy.deepcopy(FCi)
		FCo_prev = copy.deepcopy(FCo)
		FCf_prev = copy.deepcopy(FCf)

		Boo_prev = copy.deepcopy(Boo)
		Bm_prev = copy.deepcopy(Bm)
		Bi_prev = copy.deepcopy(Bi)
		Bo_prev = copy.deepcopy(Bo)
		Bf_prev = copy.deepcopy(Bf)

		FCm2_prev = copy.deepcopy(FCm2)
		FCi2_prev = copy.deepcopy(FCi2)
		FCo2_prev = copy.deepcopy(FCo2)
		FCf2_prev = copy.deepcopy(FCf2)
		
		Boo2_prev = copy.deepcopy(Boo2)
		Bm2_prev = copy.deepcopy(Bm2)
		Bi2_prev = copy.deepcopy(Bi2)
		Bo2_prev = copy.deepcopy(Bo2)
		Bf2_prev = copy.deepcopy(Bf2)

		FCm3_prev = copy.deepcopy(FCm3)
		FCi3_prev = copy.deepcopy(FCi3)
		FCo3_prev = copy.deepcopy(FCo3)
		FCf3_prev = copy.deepcopy(FCf3)
		
		Boo3_prev = copy.deepcopy(Boo3)
		Bm3_prev = copy.deepcopy(Bm3)
		Bi3_prev = copy.deepcopy(Bi3)
		Bo3_prev = copy.deepcopy(Bo3)
		Bf3_prev = copy.deepcopy(Bf3)

		FL_prev = copy.deepcopy(FL)
		
		set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_PREV)
		set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_PREV)
		set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_PREV)
	
	if step % SAVE_FREQ == 0:
		r_total_plot.append(r_total)
		err_plot.append(err)
		
		savemat(file_name, {'F1': F1, 'r_total_plot': r_total_plot, 'F2': F2, 'F3': F3, 'FL':FL, 'F1_init': F1_init, 'step': step, 'img': img,\
				'err_plot': err_plot, 'CEC':CEC, 'imgs_mean_player': imgs_mean_player,\
				'imgs_mean_red': imgs_mean_red,\
				'action_recent': action_recent, 'r_recent':r_recent,'SAVE_FREQ':SAVE_FREQ,\
				'imgs_recent': imgs_recent, 'FCi': FCi, 'FCm': FCm, 'FCf': FCf, 'FCo': FCo, 'EPS':EPS, 'NETWORK_UPDATE':NETWORK_UPDATE,\
				'FCi2': FCi2, 'FCm2': FCm2, 'FCf2': FCf2, 'FCo2': FCo2,\
				'FCi3': FCi3, 'FCm3': FCm3, 'FCf3': FCf3, 'FCo3': FCo3,'EPS_GREED_FINAL_TIME':EPS_GREED_FINAL_TIME})
		
		conv_output1 = return_buffer(CONV_OUTPUT1, gpu=GPU_CUR)
		conv_output2 = return_buffer(CONV_OUTPUT2, gpu=GPU_CUR)
		conv_output3 = return_buffer(CONV_OUTPUT3, gpu=GPU_CUR)
		print '---------------------------------------------'
		print step, 'err:',err, 'r:',r_total, 'updates:',network_updates, 'eps:', CHANCE_RAND, 't:',time.time() - t_start, file_name
		ft = EPS
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
		print 'FCm2: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCm2), np.max(FCm2), np.median(np.abs(ft*dFCm2.ravel())/np.abs(FCm2.ravel())),\
						np.min(FCm2_output), np.max(FCm2_output), np.median(FCm2_output))
		print 'FCf2: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCf2), np.max(FCf2), np.median(np.abs(ft*dFCf2.ravel())/np.abs(FCf2.ravel())),\
						np.min(FCf2_output), np.max(FCf2_output), np.median(FCf2_output))
		print 'FCi2: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCi2), np.max(FCi2), np.median(np.abs(ft*dFCi2.ravel())/np.abs(FCi2.ravel())),\
						np.min(FCi2_output), np.max(FCi2_output), np.median(FCi2_output))
		print 'FCo2: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCo2), np.max(FCo2), np.median(np.abs(ft*dFCo2.ravel())/np.abs(FCo2.ravel())),\
						np.min(FCo2_output), np.max(FCo2_output), np.median(FCo2_output))
		print 'CEC2: %f %f, kept: %f %f, new: %f %f' % (np.min(CEC2), np.max(CEC2), np.min(CEC_kept2), np.max(CEC_kept2), \
						np.min(CEC_new2), np.max(CEC_new2))
		print 'FC2.: %f %f' % (np.min(FC2_output), np.max(FC2_output))
		print
		print 'FC3m: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCm3), np.max(FCm3), np.median(np.abs(ft*dFCm3.ravel())/np.abs(FCm3.ravel())),\
						np.min(FCm3_output), np.max(FCm3_output), np.median(FCm3_output))
		print 'FC3f: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCf3), np.max(FCf3), np.median(np.abs(ft*dFCf3.ravel())/np.abs(FCf3.ravel())),\
						np.min(FCf3_output), np.max(FCf3_output), np.median(FCf3_output))
		print 'FC3i: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCi3), np.max(FCi3), np.median(np.abs(ft*dFCi3.ravel())/np.abs(FCi3.ravel())),\
						np.min(FCi3_output), np.max(FCi3_output), np.median(FCi3_output))
		print 'FC3o: %f %f (%f)   layer: %f %f (%f)' % (np.min(FCo3), np.max(FCo3), np.median(np.abs(ft*dFCo3.ravel())/np.abs(FCo3.ravel())),\
						np.min(FCo3_output), np.max(FCo3_output), np.median(FCo3_output))
		
		print 'FC3.: %f %f' % (np.min(FC3_output), np.max(FC3_output))
		print
		print 'FL: %f %f (%f)   layer: %f %f' % (np.min(FL), np.max(FL), np.median(np.abs(ft*dFL.ravel())/np.abs(FL.ravel())),\
						np.min(pred), np.max(pred))

		
		err = 0
		r_total = 0
		
		t_start = time.time()
	step += 1
	
	