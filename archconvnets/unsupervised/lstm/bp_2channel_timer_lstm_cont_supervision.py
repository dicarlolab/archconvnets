from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random

SAVE_FREQ = 5000

BATCH_SZ = 5000

FL_scale = .1
FL2_scale = .1
CEC_SCALE = 0.01

EPS_E = 4
EPS = 1*10**(-EPS_E)

n_in = 2
n1 = 128
n2 = 128

np.random.seed(6166)

FCm = np.single(np.random.normal(scale=FL_scale, size=(n1, n_in)))
FCi = np.single(np.random.normal(scale=FL_scale, size=(n1, n_in)))
FCo = np.single(np.random.normal(scale=FL_scale, size=(n1, n_in)))
FCf = np.single(np.random.normal(scale=FL_scale, size=(n1, n_in)))# - .1
CEC = np.single(np.random.normal(scale=CEC_SCALE, size=n1))

FC2f = np.single(np.random.normal(scale=FL2_scale, size=(n2, n1)))# - .1
FC2o = np.single(np.random.normal(scale=FL2_scale, size=(n2, n1)))
FC2i = np.single(np.random.normal(scale=FL2_scale, size=(n2, n1)))
FC2m = np.single(np.random.normal(scale=FL2_scale, size=(n2, n1)))
CEC2 = np.single(np.random.normal(scale=CEC_SCALE, size=n2))

CEC2_dFC2m = np.zeros_like(CEC2)
CEC2_dFC2i = np.zeros_like(CEC2)
CEC2_dFC2f = np.zeros_like(CEC2)

CEC_dFCm = np.zeros_like(CEC)
CEC_dFCi = np.zeros_like(CEC)
CEC_dFCf = np.zeros_like(CEC)

FL = np.single(np.random.normal(scale=1, size=n2))

CEC_kept = 0; CEC_new = 0
CEC2_kept = 0; CEC2_new = 0

dFL = np.zeros_like(FL)
dFCm = np.zeros_like(FCm)
dFCi = np.zeros_like(FCi)
dFCo = np.zeros_like(FCo)
dFCf = np.zeros_like(FCf)
dFC2m = np.zeros_like(FC2m)
dFC2i = np.zeros_like(FC2i)
dFC2o = np.zeros_like(FC2o)
dFC2f = np.zeros_like(FC2f)

t_start = time.time()

MAX_TIME = 20

time_length = np.random.randint(MAX_TIME-1) + 1
elapsed_time = 0
p_start = .05

inputs = np.zeros(n_in)

inputs_recent = np.zeros((SAVE_FREQ, n_in))
targets_recent = np.zeros(SAVE_FREQ)
preds_recent = np.zeros(SAVE_FREQ)

err_t = 0
err = []

global_step = 0
while True:
	if random.random() < p_start and elapsed_time > time_length:
		time_length = np.random.randint(MAX_TIME)
		elapsed_time = 0
		inputs[0] = 1
		inputs[1] = time_length
	else:
		inputs[0] = 0
		
	target = 1 - (time_length < elapsed_time)
	
	# forward pass
	FCm_output_pre = np.dot(FCm, inputs)
	FCi_output_pre = np.dot(FCi, inputs)
	FCo_output_pre = np.dot(FCo, inputs)
	FCf_output_pre = np.dot(FCf, inputs)
	
	FCf_output = 1 / (1 + np.exp(-FCf_output_pre))
	FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
	FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
	FCm_output = 1 / (1 + np.exp(-FCm_output_pre))
	#FCm_output = FCm_output_pre
	
	FC_output = FCo_output * (FCf_output * CEC + FCi_output * FCm_output)
	
	CEC_kept = CEC*FCf_output
	CEC_new = FCi_output*FCm_output
	
	CEC = CEC*FCf_output + FCi_output*FCm_output
	
	FC2f_output_pre = np.dot(FC2f, FC_output)
	FC2o_output_pre = np.dot(FC2o, FC_output)
	FC2i_output_pre = np.dot(FC2i, FC_output)
	FC2m_output_pre = np.dot(FC2m, FC_output)
	
	FC2f_output = 1 / (1 + np.exp(-FC2f_output_pre))
	FC2o_output = 1 / (1 + np.exp(-FC2o_output_pre))
	FC2i_output = 1 / (1 + np.exp(-FC2i_output_pre))
	FC2m_output = 1 / (1 + np.exp(-FC2m_output_pre))
	#FC2m_output = FC2m_output_pre
	
	FC2_output = FC2o_output * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	CEC2_kept = CEC2*FC2f_output
	CEC2_new = FC2i_output*FC2m_output
	
	CEC2 = CEC2*FC2f_output + FC2i_output*FC2m_output
	
	pred = np.dot(FL, FC2_output)
	
	pred_m_Y = pred - target
	
	err_t += pred_m_Y**2
	
	############### reverse pointwise

	FC2f_output_rev = np.exp(FC2f_output_pre)/((np.exp(FC2f_output_pre) + 1)**2)
	FC2o_output_rev = np.exp(FC2o_output_pre)/((np.exp(FC2o_output_pre) + 1)**2)
	FC2i_output_rev = np.exp(FC2i_output_pre)/((np.exp(FC2i_output_pre) + 1)**2)
	FC2m_output_rev = np.exp(FC2m_output_pre)/((np.exp(FC2m_output_pre) + 1)**2)
	#FC2m_output_rev = 1
	
	FCf_output_rev = np.exp(FCf_output_pre)/((np.exp(FCf_output_pre) + 1)**2)
	FCi_output_rev = np.exp(FCi_output_pre)/((np.exp(FCi_output_pre) + 1)**2)
	FCo_output_rev = np.exp(FCo_output_pre)/((np.exp(FCo_output_pre) + 1)**2)
	FCm_output_rev = np.exp(FCm_output_pre)/((np.exp(FCm_output_pre) + 1)**2)
	#FCm_output_rev = 1
	
	
	############ FL
	
	dFL = pred_m_Y * FC2_output
	
	above_w = pred_m_Y * FL
	
	######################### mem 2 gradients:
	
	FC2f_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2f * FC2f_output + FC2f_output_rev * CEC2)
	FC2i_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2i * FC2f_output + FC2i_output_rev * FC2m_output)
	FC2m_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2m * FC2f_output + FC2i_output * FC2m_output_rev)
	FC2o_output_rev_sig = above_w * FC2o_output_rev * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	CEC2_dFC2f = CEC2_dFC2f * FC2f_output + FC2f_output_rev * CEC2
	CEC2_dFC2m = CEC2_dFC2m * FC2f_output + FC2i_output * FC2m_output_rev
	CEC2_dFC2i = CEC2_dFC2i * FC2f_output + FC2i_output_rev * FC2m_output
	
	dFC2f += np.einsum(FC_output, [0], FC2f_output_rev_sig, [1], [1,0])
	dFC2i += np.einsum(FC_output, [0], FC2i_output_rev_sig, [1], [1,0])
	dFC2m += np.einsum(FC_output, [0], FC2m_output_rev_sig, [1], [1,0])
	dFC2o += np.einsum(FC_output, [0], FC2o_output_rev_sig, [1], [1,0])
	
	above_w = np.einsum(FC2o, [0,1], FC2o_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2f, [0,1], FC2f_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2i, [0,1], FC2i_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2m, [0,1], FC2m_output_rev_sig, [0], [1])
	
	
	########################## mem 1 gradients:

	FCf_output_rev_sig = above_w * FCo_output * (CEC_dFCf * FCf_output + FCf_output_rev * CEC)
	FCi_output_rev_sig = above_w * FCo_output * (CEC_dFCi * FCf_output + FCi_output_rev * FCm_output)
	FCm_output_rev_sig = above_w * FCo_output * (CEC_dFCm * FCf_output + FCi_output * FCm_output_rev)
	FCo_output_rev_sig = above_w * FCo_output_rev * (FCf_output * CEC + FCi_output * FCm_output)
	
	CEC_dFCf = CEC_dFCf * FCf_output + FCf_output_rev * CEC
	CEC_dFCm = CEC_dFCm * FCf_output + FCi_output * FCm_output_rev
	CEC_dFCi = CEC_dFCi * FCf_output + FCi_output_rev * FCm_output
	
	dFCf += np.einsum(inputs, [0], FCf_output_rev_sig, [1], [1,0])
	dFCi += np.einsum(inputs, [0], FCi_output_rev_sig, [1], [1,0])
	dFCm += np.einsum(inputs, [0], FCm_output_rev_sig, [1], [1,0])
	dFCo += np.einsum(inputs, [0], FCo_output_rev_sig, [1], [1,0])
	
	if global_step % BATCH_SZ == 0:
		FCf -= dFCf*EPS/BATCH_SZ
		FCo -= dFCo*EPS/BATCH_SZ
		FCm -= dFCm*EPS/BATCH_SZ
		FCi -= dFCi*EPS/BATCH_SZ
		
		FC2f -= dFC2f*EPS/BATCH_SZ
		FC2o -= dFC2o*EPS/BATCH_SZ
		FC2m -= dFC2m*EPS/BATCH_SZ
		FC2i -= dFC2i*EPS/BATCH_SZ
		
		FL -= dFL*EPS/BATCH_SZ
		
		dFCf = np.zeros_like(FCf)
		dFCo = np.zeros_like(FCo)
		dFCm = np.zeros_like(FCm)
		dFCi = np.zeros_like(FCi)
		
		dFC2f = np.zeros_like(FC2f)
		dFC2o = np.zeros_like(FC2o)
		dFC2m = np.zeros_like(FC2m)
		dFC2i = np.zeros_like(FC2i)
	
	s_loc = global_step % SAVE_FREQ
	
	inputs_recent[s_loc] = copy.deepcopy(inputs)
	targets_recent[s_loc] = target
	preds_recent[s_loc] = pred
	
	if global_step % SAVE_FREQ == 0:
		err.append(err_t)
		print '---------------------------------------------'
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
		savemat('/home/darren/timer.mat', {'inputs_recent': inputs_recent, 'targets_recent': targets_recent, \
			'preds_recent': preds_recent, 'err': err})
	
		t_start = time.time()
		err_t = 0
	
	global_step += 1
	elapsed_time += 1
		
