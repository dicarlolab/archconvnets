from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random

BATCH_SZ = 1#00

filename = '/home/darren/timer_linear_' + str(BATCH_SZ) + 'batch2.mat'
SAVE_FREQ = 256*2*2*2*4*2

N_TRAIN = 2000000#140902#4

FL_scale = .01
FL2_scale = .05
FL3_scale = .05
CEC_SCALE = 0.001

EPS_E = 3
EPS = 1*10**(-EPS_E)

n_in = 2
n1 = 2*128#5#2*2*256#*4*2
n2 = 1+2*256#*4*2
n3 = 2+2*256#*4*2

np.random.seed(6166)

## l1
FCm = 2*np.single(np.random.random(size=(n1, n_in))) - 1
FCi = 2*np.single(np.random.random(size=(n1, n_in))) - 1
FCo = 2*np.single(np.random.random(size=(n1, n_in))) - 1
FCf = 2*np.single(np.random.random(size=(n1, n_in))) - 1

Bm = np.zeros_like(2*np.single(np.random.random(size=n1)) - 1) ##
Bi = np.zeros_like(2*np.single(np.random.random(size=n1)) - 1)
Bo = 2*np.ones_like(2*np.single(np.random.random(size=n1)) - 1)
Boo = np.zeros_like(2*np.single(np.random.random(size=n1)) - 1) ##
Bf = -2*np.ones_like(2*np.single(np.random.random(size=n1)) - 1)
CEC = np.zeros_like(2*np.single(np.random.random(size=n1)) - 1)


CEC_dFCm = np.zeros_like(CEC)
CEC_dFCi = np.zeros_like(CEC)
CEC_dFCf = np.zeros_like(CEC)

FL = np.single(np.random.normal(scale=1, size=n1))

CEC_kept = 0; CEC_new = 0


dFL = np.zeros_like(FL)

# l1
dFCm = np.zeros_like(FCm)
dFCi = np.zeros_like(FCi)
dFCo = np.zeros_like(FCo)
dFCf = np.zeros_like(FCf)

dBm = np.zeros_like(Bm)
dBi = np.zeros_like(Bi)
dBo = np.zeros_like(Bo)
dBoo = np.zeros_like(Boo)
dBf = np.zeros_like(Bf)


t_start = time.time()

MAX_TIME = 4#0

time_length = np.random.randint(MAX_TIME-1) + 1
elapsed_time = 0
p_start = .1

inputs = np.zeros(n_in)

inputs_recent = np.zeros((SAVE_FREQ, n_in))
targets_recent = np.zeros(SAVE_FREQ)
preds_recent = np.zeros(SAVE_FREQ)

err_t = 0
err = []

global_step = 0
while True:
	if random.random() < p_start and elapsed_time > time_length:
		time_length = np.random.randint(6)
		elapsed_time = 0
		inputs[0] = 1
		inputs[1] = time_length/3.
	else:
		inputs[0] = 0
		
	target = 1 - (time_length < elapsed_time)
	
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
	
	pred = np.dot(FL, FC_output)
	
	pred_m_Y = pred - target
	
	err_t += pred_m_Y**2
	
	############### reverse pointwise
	FCf_output_rev = np.exp(FCf_output_pre)/((np.exp(FCf_output_pre) + 1)**2)
	FCi_output_rev = np.exp(FCi_output_pre)/((np.exp(FCi_output_pre) + 1)**2)
	FCo_output_rev = np.exp(FCo_output_pre)/((np.exp(FCo_output_pre) + 1)**2)
	FCm_output_rev = np.exp(FCm_output_pre)/((np.exp(FCm_output_pre) + 1)**2)
	
	############ FL
	
	dFL = pred_m_Y * FC_output
	
	above_w = pred_m_Y * FL
	
	dBoo += above_w
	
	########################## mem 1 gradients:

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
	
	if global_step % SAVE_FREQ == 0:
		err.append(err_t)
		print '---------------------------------------------', filename
		if global_step < N_TRAIN:
			print 'update'
		print global_step, 'err:', err[-1], time.time() - t_start
		ft = EPS
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
		
		savemat(filename, {'inputs_recent': inputs_recent, 'targets_recent': targets_recent, \
			'preds_recent': preds_recent, 'err': err})
	
		t_start = time.time()
		err_t = 0
	
	if global_step % BATCH_SZ == 0 and global_step < N_TRAIN:
		FCf -= dFCf*EPS/BATCH_SZ
		FCo -= dFCo*EPS/BATCH_SZ
		FCm -= dFCm*EPS/BATCH_SZ
		FCi -= dFCi*EPS/BATCH_SZ
		
		Bf -= dBf*EPS/BATCH_SZ
		Bo -= dBo*EPS/BATCH_SZ
		Bm -= dBm*EPS/BATCH_SZ
		Bi -= dBi*EPS/BATCH_SZ
		Boo -= dBoo*EPS/BATCH_SZ
		
		FL -= dFL*EPS/BATCH_SZ

	
	s_loc = global_step % SAVE_FREQ
	
	inputs_recent[s_loc] = copy.deepcopy(inputs)
	targets_recent[s_loc] = target
	preds_recent[s_loc] = pred
	
	global_step += 1
	elapsed_time += 1
		
