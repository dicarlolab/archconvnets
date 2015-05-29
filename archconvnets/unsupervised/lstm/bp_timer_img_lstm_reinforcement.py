from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random

use_time = True
if use_time:
	filename = '/home/darren/timer_linear_1024.mat'
	GPU_CUR = 0
	GPU_PREV = 1
else:
	filename = '/home/darren/timer_linear_nt.mat'
	GPU_CUR = 2
	GPU_PREV = 3

EPS_GREED_FINAL = .1
EPS_GREED_FINAL_TIME = 3*5*2000000/150#4#200000*3
SAVE_FREQ = 1000
GAMMA = 0.99
BATCH_SZ = 1
NETWORK_UPDATE = 10000

BATCH_SZ = 1

F1_scale = .1
FL_scale = .01
FL2_scale = .01
FL3_scale = .01
CEC_SCALE = 0.001

EPS_E = 5
EPS = 5*10**(-EPS_E)

EPS_E = 4
EPS_F = 1*10**(-EPS_E)

n_in = 2
n1 = 128#1024#32#2*128#*4*2
n2 = 128#1024#32#2*128 +2#+ 2#*4*2
n3 = 128#1024#32#2*128 +1#+ 3#*4*2

n1f = 32
s1 = 5


# gpu buffer indices
MAX_OUTPUT1 = 0; DF2_DATA = 1; CONV_OUTPUT1 = 2; DPOOL1 = 3
F1_IND = 4; IMGS_PAD = 5; DF1 = 6; F2_IND = 11
D_UNPOOL2 = 12;F3_IND = 13; MAX_OUTPUT2 = 14; MAX_OUTPUT3 = 15
CONV_OUTPUT1 = 19; CONV_OUTPUT2 = 20; CONV_OUTPUT3 = 21
DF2 = 25; DPOOL2 = 26; DF3_DATA = 27; DPOOL3 = 28; DF3 = 29; FL_PRED = 30;
FL_IND = 31; PRED = 32; DFL = 33

np.random.seed(6166)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1f, 3, s1, s1)))

max_output_sz1 = 16

## l1
FCm = np.single(np.random.normal(scale=FL_scale, size=(n1, n1f, max_output_sz1, max_output_sz1)))
FCi = np.single(np.random.normal(scale=FL_scale, size=(n1, n1f, max_output_sz1, max_output_sz1)))
FCo = np.single(np.random.normal(scale=FL_scale, size=(n1, n1f, max_output_sz1, max_output_sz1)))
FCf = np.single(np.random.normal(scale=FL_scale, size=(n1, n1f, max_output_sz1, max_output_sz1)))

Bm = np.single(np.random.normal(scale=FL_scale, size=n1))
Bi = np.single(np.random.normal(scale=FL_scale, size=n1))
Bo = np.single(np.random.normal(scale=FL_scale, size=n1))
Bf = np.single(np.random.normal(scale=FL_scale, size=n1)) #- 3
CEC = np.single(np.random.normal(scale=CEC_SCALE, size=n1))

## l2
FC2f = np.single(np.random.normal(scale=FL2_scale, size=(n2, n1)))
FC2o = np.single(np.random.normal(scale=FL2_scale, size=(n2, n1)))
FC2i = np.single(np.random.normal(scale=FL2_scale, size=(n2, n1)))
FC2m = np.single(np.random.normal(scale=FL2_scale, size=(n2, n1)))

B2m = np.single(np.random.normal(scale=FL2_scale, size=n2))
B2i = np.single(np.random.normal(scale=FL2_scale, size=n2))
B2o = np.single(np.random.normal(scale=FL2_scale, size=n2))
B2f = np.single(np.random.normal(scale=FL2_scale, size=n2)) #- 3
CEC2 = np.single(np.random.normal(scale=CEC_SCALE, size=n2))

### l3
FC3f = np.single(np.random.normal(scale=FL3_scale, size=(n3, n2)))
FC3o = np.single(np.random.normal(scale=FL3_scale, size=(n3, n2)))
FC3i = np.single(np.random.normal(scale=FL3_scale, size=(n3, n2)))
FC3m = np.single(np.random.normal(scale=FL3_scale, size=(n3, n2)))

B3m = np.single(np.random.normal(scale=FL3_scale, size=n3))
B3i = np.single(np.random.normal(scale=FL3_scale, size=n3))
B3o = np.single(np.random.normal(scale=FL3_scale, size=n3))
B3f = np.single(np.random.normal(scale=FL3_scale, size=n3)) #- 3
CEC3 = np.single(np.random.normal(scale=CEC_SCALE, size=n3))

CEC3_dFC3m = np.zeros_like(CEC3)
CEC3_dFC3i = np.zeros_like(CEC3)
CEC3_dFC3f = np.zeros_like(CEC3)

CEC2_dFC2m = np.zeros_like(CEC2)
CEC2_dFC2i = np.zeros_like(CEC2)
CEC2_dFC2f = np.zeros_like(CEC2)

CEC_dFCm = np.zeros_like(CEC)
CEC_dFCi = np.zeros_like(CEC)
CEC_dFCf = np.zeros_like(CEC)

FL = np.single(np.random.normal(scale=5, size=(2,n3)))

CEC_kept = 0; CEC_new = 0
CEC2_kept = 0; CEC2_new = 0
CEC3_kept = 0; CEC3_new = 0


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
set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)

t_start = time.time()

MAX_TIME = 70

time_length = np.random.randint(MAX_TIME-1) + 1
elapsed_time = 0
p_start = .05

inputs = np.zeros(n_in)

inputs_recent = np.zeros((SAVE_FREQ, n_in))
targets_recent = np.zeros(SAVE_FREQ)
preds_recent = np.zeros(SAVE_FREQ)

err = 0
err_plot = []
r_total_plot = []
r_total = 0

global_step = 0
network_updates = 0

# show blocks
img = np.zeros((1,3,32,32),dtype='single')
if random.random() < p_start and elapsed_time > time_length:
	time_length = np.random.randint(MAX_TIME)
	elapsed_time = 0
	img[0,1,:4,:4] = 1 # start signal
	img[0,0,28:,28:] = time_length
else:
	img[0,1,:4,:4] = 0 # start signal
	
target = 1 - (time_length < elapsed_time)

while True:
	# forward pass
	set_buffer(img, IMGS_PAD, gpu=GPU_CUR)
	
	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_CUR)
	
	max_output1 = return_buffer(MAX_OUTPUT1, gpu=GPU_CUR)
	
	FCm_output_pre = np.einsum(FCm, range(4), max_output1, [4, 1,2,3], [4, 0]) + Bm
	FCi_output_pre = np.einsum(FCi, range(4), max_output1, [4, 1,2,3], [4, 0]) + Bi
	FCo_output_pre = np.einsum(FCo, range(4), max_output1, [4, 1,2,3], [4, 0]) + Bo
	FCf_output_pre = np.einsum(FCf, range(4), max_output1, [4, 1,2,3], [4, 0]) + Bf
	
	FCf_output = 1 / (1 + np.exp(-FCf_output_pre))
	FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
	FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
	FCm_output = FCm_output_pre
	
	FC_output = FCo_output * (FCf_output * CEC + FCi_output * FCm_output)
	
	CEC_kept = CEC*FCf_output
	CEC_new = FCi_output*FCm_output
	
	if use_time:
		CEC = CEC*FCf_output + FCi_output*FCm_output
	
	FC2f_output_pre = np.dot(FC2f, np.squeeze(FC_output)) + B2f
	FC2o_output_pre = np.dot(FC2o, np.squeeze(FC_output)) + B2o
	FC2i_output_pre = np.dot(FC2i, np.squeeze(FC_output)) + B2i
	FC2m_output_pre = np.dot(FC2m, np.squeeze(FC_output)) + B2m
	
	FC2f_output = 1 / (1 + np.exp(-FC2f_output_pre))
	FC2o_output = 1 / (1 + np.exp(-FC2o_output_pre))
	FC2i_output = 1 / (1 + np.exp(-FC2i_output_pre))
	FC2m_output = FC2m_output_pre
	
	FC2_output = FC2o_output * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	CEC2_kept = CEC2*FC2f_output
	CEC2_new = FC2i_output*FC2m_output
	
	if use_time:
		CEC2 = CEC2*FC2f_output + FC2i_output*FC2m_output
	
	FC3f_output_pre = np.dot(FC3f, FC2_output) + B3f
	FC3o_output_pre = np.dot(FC3o, FC2_output) + B3o
	FC3i_output_pre = np.dot(FC3i, FC2_output) + B3i
	FC3m_output_pre = np.dot(FC3m, FC2_output) + B3m
	
	FC3f_output = 1 / (1 + np.exp(-FC3f_output_pre))
	FC3o_output = 1 / (1 + np.exp(-FC3o_output_pre))
	FC3i_output = 1 / (1 + np.exp(-FC3i_output_pre))
	FC3m_output = FC3m_output_pre
	
	FC3_output = FC3o_output * (FC3f_output * CEC3 + FC3i_output * FC3m_output)
	
	CEC3_kept = CEC3*FC3f_output
	CEC3_new = FC3i_output*FC3m_output
	
	if use_time:
		CEC3 = CEC3*FC3f_output + FC3i_output*FC3m_output
	
	pred = np.dot(FL, FC3_output)
	
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
	CHANCE_RAND = np.max((1 - ((1-EPS_GREED_FINAL)/EPS_GREED_FINAL_TIME)*global_step, EPS_GREED_FINAL))
	if np.random.rand() <= CHANCE_RAND:
		action = np.random.randint(2)
	else:
		action = np.argmax(pred)

	r = 2*(((target - action)**2) < .01) - 1
	r_total += r
	
	# create next inputs/targets
	img = np.zeros((1,3,32,32),dtype='single')
	if random.random() < p_start and elapsed_time > time_length:
		time_length = np.random.randint(MAX_TIME)
		elapsed_time = 0
		img[0,1,:4,:4] = 1 # start signal
		img[0,0,28:,28:] = time_length
	else:
		img[0,1,:4,:4] = 0 # start signal
		
	target = 1 - (time_length < elapsed_time)
	
	# forward pass prev network
	set_buffer(img, IMGS_PAD, gpu=GPU_PREV)
	conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_PREV)
	max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_PREV)
	
	# compute target
	max_output1_prev = return_buffer(MAX_OUTPUT1, gpu=GPU_PREV)
	
	FCm_output_pre = np.einsum(FCm_prev, range(4), max_output1_prev, [4, 1,2,3], [4, 0]) + Bm_prev
	FCi_output_pre = np.einsum(FCi_prev, range(4), max_output1_prev, [4, 1,2,3], [4, 0]) + Bi_prev
	FCo_output_pre = np.einsum(FCo_prev, range(4), max_output1_prev, [4, 1,2,3], [4, 0]) + Bo_prev
	FCf_output_pre = np.einsum(FCf_prev, range(4), max_output1_prev, [4, 1,2,3], [4, 0]) + Bf_prev
	
	FCf_output_prev = 1 / (1 + np.exp(-FCf_output_pre))
	FCi_output_prev = 1 / (1 + np.exp(-FCi_output_pre))
	FCo_output_prev = 1 / (1 + np.exp(-FCo_output_pre))
	FCm_output_prev = FCm_output_pre
	
	FC_output_prev = FCo_output_prev * (FCf_output_prev * CEC_prev + FCi_output_prev * FCm_output_prev)
	
	if use_time:
		CEC_prev = CEC_prev*FCf_output_prev + FCi_output_prev*FCm_output_prev
	
	FC2f_output_pre = np.dot(FC2f_prev, np.squeeze(FC_output_prev)) + B2f_prev
	FC2o_output_pre = np.dot(FC2o_prev, np.squeeze(FC_output_prev)) + B2o_prev
	FC2i_output_pre = np.dot(FC2i_prev, np.squeeze(FC_output_prev)) + B2i_prev
	FC2m_output_pre = np.dot(FC2m_prev, np.squeeze(FC_output_prev)) + B2m_prev
	
	FC2f_output_prev = 1 / (1 + np.exp(-FC2f_output_pre))
	FC2o_output_prev = 1 / (1 + np.exp(-FC2o_output_pre))
	FC2i_output_prev = 1 / (1 + np.exp(-FC2i_output_pre))
	FC2m_output_prev = FC2m_output_pre
	
	FC2_output_prev = FC2o_output_prev * (FC2f_output_prev * CEC2_prev + FC2i_output_prev * FC2m_output_prev)
	
	if use_time:
		CEC2_prev = CEC2_prev*FC2f_output_prev + FC2i_output_prev*FC2m_output_prev
	
	FC3f_output_pre = np.dot(FC3f_prev, FC2_output) + B3f_prev
	FC3o_output_pre = np.dot(FC3o_prev, FC2_output) + B3o_prev
	FC3i_output_pre = np.dot(FC3i_prev, FC2_output) + B3i_prev
	FC3m_output_pre = np.dot(FC3m_prev, FC2_output) + B3m_prev
	
	FC3f_output_prev = 1 / (1 + np.exp(-FC3f_output_pre))
	FC3o_output_prev = 1 / (1 + np.exp(-FC3o_output_pre))
	FC3i_output_prev = 1 / (1 + np.exp(-FC3i_output_pre))
	FC3m_output_prev = FC3m_output_pre
	
	FC3_output_prev = FC3o_output_prev * (FC3f_output_prev * CEC3_prev + FC3i_output_prev * FC3m_output_prev)
	
	if use_time:
		CEC3_prev = CEC3_prev*FC3f_output_prev + FC3i_output_prev*FC3m_output_prev
	
	pred_prev = np.dot(FL_prev, FC3_output_prev)
	
	y_output = np.single(r + GAMMA * np.max(pred_prev))
	
	############################ backprop
	
	pred_m_Y = y_output - pred[action]
	
	err += pred_m_Y**2
	
	############ FL
	
	dFL = np.zeros_like(FL)
	dFL[action] = pred_m_Y * FC3_output
	
	above_w = pred_m_Y * FL[action]
	
	######################### mem 3 gradients:
	
	FC3f_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3f * FC3f_output + FC3f_output_rev * CEC3)
	FC3i_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3i * FC3f_output + FC3i_output_rev * FC3m_output)
	FC3m_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3m * FC3f_output + FC3i_output * FC3m_output_rev)
	FC3o_output_rev_sig = above_w * FC3o_output_rev * (FC3f_output * CEC3 + FC3i_output * FC3m_output)
	
	if use_time:
		CEC3_dFC3f = CEC3_dFC3f * FC3f_output + FC3f_output_rev * CEC3
		CEC3_dFC3m = CEC3_dFC3m * FC3f_output + FC3i_output * FC3m_output_rev
		CEC3_dFC3i = CEC3_dFC3i * FC3f_output + FC3i_output_rev * FC3m_output
	
	dB3f = np.squeeze(FC3f_output_rev_sig)
	dB3i = np.squeeze(FC3i_output_rev_sig)
	dB3m = np.squeeze(FC3m_output_rev_sig)
	dB3o = np.squeeze(FC3o_output_rev_sig)
	
	dFC3f = np.einsum(FC2_output, [0], FC3f_output_rev_sig, [1], [1,0])
	dFC3i = np.einsum(FC2_output, [0], FC3i_output_rev_sig, [1], [1,0])
	dFC3m = np.einsum(FC2_output, [0], FC3m_output_rev_sig, [1], [1,0])
	dFC3o = np.einsum(FC2_output, [0], FC3o_output_rev_sig, [1], [1,0])
	
	above_w = np.einsum(FC3o, [0,1], FC3o_output_rev_sig, [0], [1])
	above_w += np.einsum(FC3f, [0,1], FC3f_output_rev_sig, [0], [1])
	above_w += np.einsum(FC3i, [0,1], FC3i_output_rev_sig, [0], [1])
	above_w += np.einsum(FC3m, [0,1], FC3m_output_rev_sig, [0], [1])
	
	######################### mem 2 gradients:
	
	FC2f_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2f * FC2f_output + FC2f_output_rev * CEC2)
	FC2i_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2i * FC2f_output + FC2i_output_rev * FC2m_output)
	FC2m_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2m * FC2f_output + FC2i_output * FC2m_output_rev)
	FC2o_output_rev_sig = above_w * FC2o_output_rev * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	if use_time:
		CEC2_dFC2f = CEC2_dFC2f * FC2f_output + FC2f_output_rev * CEC2
		CEC2_dFC2m = CEC2_dFC2m * FC2f_output + FC2i_output * FC2m_output_rev
		CEC2_dFC2i = CEC2_dFC2i * FC2f_output + FC2i_output_rev * FC2m_output
	
	dB2f = np.squeeze(FC2f_output_rev_sig)
	dB2i = np.squeeze(FC2i_output_rev_sig)
	dB2m = np.squeeze(FC2m_output_rev_sig)
	dB2o = np.squeeze(FC2o_output_rev_sig)
	
	dFC2f = np.einsum(np.squeeze(FC_output), [0], FC2f_output_rev_sig, [1], [1,0])
	dFC2i = np.einsum(np.squeeze(FC_output), [0], FC2i_output_rev_sig, [1], [1,0])
	dFC2m = np.einsum(np.squeeze(FC_output), [0], FC2m_output_rev_sig, [1], [1,0])
	dFC2o = np.einsum(np.squeeze(FC_output), [0], FC2o_output_rev_sig, [1], [1,0])
	
	above_w = np.einsum(FC2o, [0,1], FC2o_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2f, [0,1], FC2f_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2i, [0,1], FC2i_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2m, [0,1], FC2m_output_rev_sig, [0], [1])
	
	
	########################## mem 1 gradients:

	FCf_output_rev_sig = above_w * FCo_output * (CEC_dFCf * FCf_output + FCf_output_rev * CEC)
	FCi_output_rev_sig = above_w * FCo_output * (CEC_dFCi * FCf_output + FCi_output_rev * FCm_output)
	FCm_output_rev_sig = above_w * FCo_output * (CEC_dFCm * FCf_output + FCi_output * FCm_output_rev)
	FCo_output_rev_sig = above_w * FCo_output_rev * (FCf_output * CEC + FCi_output * FCm_output)
	
	if use_time:
		CEC_dFCf = CEC_dFCf * FCf_output + FCf_output_rev * CEC
		CEC_dFCm = CEC_dFCm * FCf_output + FCi_output * FCm_output_rev
		CEC_dFCi = CEC_dFCi * FCf_output + FCi_output_rev * FCm_output
	
	dBf = np.squeeze(FCf_output_rev_sig)
	dBi = np.squeeze(FCi_output_rev_sig)
	dBm = np.squeeze(FCm_output_rev_sig)
	dBo = np.squeeze(FCo_output_rev_sig)
	
	dFCf = np.einsum(max_output1, range(4), FCf_output_rev_sig, [0,4], [4,1,2,3])
	dFCi = np.einsum(max_output1, range(4), FCi_output_rev_sig, [0,4], [4,1,2,3])
	dFCm = np.einsum(max_output1, range(4), FCm_output_rev_sig, [0,4], [4,1,2,3])
	dFCo = np.einsum(max_output1, range(4), FCo_output_rev_sig, [0,4], [4,1,2,3])
	
	above_w = np.einsum(FCo, range(4), FCo_output_rev_sig, [4,0], [4,1,2,3])
	above_w += np.einsum(FCi, range(4), FCi_output_rev_sig, [4,0], [4,1,2,3])
	above_w += np.einsum(FCm, range(4), FCm_output_rev_sig, [4,0], [4,1,2,3])
	above_w += np.einsum(FCf, range(4), FCf_output_rev_sig, [4,0], [4,1,2,3])
	
	set_buffer(above_w, FL_PRED, gpu=GPU_CUR)
	
	########### backprop
	max_pool_back_cudnn_buffers(MAX_OUTPUT1, FL_PRED, CONV_OUTPUT1, DPOOL1, gpu=GPU_CUR)
	conv_dfilter_buffers(F1_IND, IMGS_PAD, DPOOL1, DF1, stream=1, gpu=GPU_CUR)
	
	dF1 = return_buffer(DF1, stream=1, gpu=GPU_CUR)
	
	#####
	FCf += dFCf*EPS
	FCo += dFCo*EPS
	FCm += dFCm*EPS
	FCi += dFCi*EPS
	
	Bf += dBf*EPS
	Bo += dBo*EPS
	Bm += dBm*EPS
	Bi += dBi*EPS
	
	FC2f += dFC2f*EPS
	FC2o += dFC2o*EPS
	FC2m += dFC2m*EPS
	FC2i += dFC2i*EPS
	
	B2f += dB2f*EPS
	B2o += dB2o*EPS
	B2m += dB2m*EPS
	B2i += dB2i*EPS
	
	FC3f += dFC3f*EPS
	FC3o += dFC3o*EPS
	FC3m += dFC3m*EPS
	FC3i += dFC3i*EPS
	
	B3f += dB3f*EPS
	B3o += dB3o*EPS
	B3m += dB3m*EPS
	B3i += dB3i*EPS
	
	FL += dFL*EPS
	
	F1 += dF1*EPS_F
	
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
	
	s_loc = global_step % SAVE_FREQ
	
	inputs_recent[s_loc] = copy.deepcopy(inputs)
	targets_recent[s_loc] = target
	preds_recent[s_loc] = action
	
	if global_step % SAVE_FREQ == 0 and global_step != 0:
		err_plot.append(err)
		r_total_plot.append(r_total)
		conv_output1 = return_buffer(CONV_OUTPUT1, gpu=GPU_CUR)
		print '---------------------------------------------', filename
		print global_step, 'err:', err_plot[-1], 'r:', r_total_plot[-1], time.time() - t_start, 'eps:', CHANCE_RAND
		ft = EPS
		print 'F1: %f %f (%f);   layer: %f %f' % (np.min(F1), np.max(F1), np.median(np.abs(ft*dF1.ravel())/np.abs(F1.ravel())),\
						np.min(conv_output1), np.max(conv_output1))
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
		savemat(filename, {'inputs_recent': inputs_recent, 'targets_recent': targets_recent, \
			'preds_recent': preds_recent, 'err': err_plot, 'r_total_plot':r_total_plot,'F1':F1})
	
		t_start = time.time()
		err = 0
		r_total = 0
	
	global_step += 1
	elapsed_time += 1
		
