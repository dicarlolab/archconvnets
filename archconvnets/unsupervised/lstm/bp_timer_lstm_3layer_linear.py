from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random

SAVE_FREQ = 500

BATCH_SZ = 1

FL_scale = .01
FL2_scale = .01
FL3_scale = .02
CEC_SCALE = 0.001

EPS_E = 4
EPS = 1*10**(-EPS_E)

n_in = 2
n1 = 128*4
n2 = 128*4
n3 = 128*4

np.random.seed(6166)

## l1
FCm = np.single(np.random.normal(scale=FL_scale, size=(n1, n_in)))
FCi = np.single(np.random.normal(scale=FL_scale, size=(n1, n_in)))
FCo = np.single(np.random.normal(scale=FL_scale, size=(n1, n_in)))
FCf = np.single(np.random.normal(scale=FL_scale, size=(n1, n_in)))

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

FL = np.single(np.random.normal(scale=5, size=n3))

CEC_kept = 0; CEC_new = 0
CEC2_kept = 0; CEC2_new = 0
CEC3_kept = 0; CEC3_new = 0

dFL = np.zeros_like(FL)

# l1
dFCm = np.zeros_like(FCm)
dFCi = np.zeros_like(FCi)
dFCo = np.zeros_like(FCo)
dFCf = np.zeros_like(FCf)

dBm = np.zeros_like(Bm)
dBi = np.zeros_like(Bi)
dBo = np.zeros_like(Bo)
dBf = np.zeros_like(Bf)

# l2
dFC2m = np.zeros_like(FC2m)
dFC2i = np.zeros_like(FC2i)
dFC2o = np.zeros_like(FC2o)
dFC2f = np.zeros_like(FC2f)

dB2m = np.zeros_like(B2m)
dB2i = np.zeros_like(B2i)
dB2o = np.zeros_like(B2o)
dB2f = np.zeros_like(B2f)

# l3
dFC3m = np.zeros_like(FC3m)
dFC3i = np.zeros_like(FC3i)
dFC3o = np.zeros_like(FC3o)
dFC3f = np.zeros_like(FC3f)

dB3m = np.zeros_like(B3m)
dB3i = np.zeros_like(B3i)
dB3o = np.zeros_like(B3o)
dB3f = np.zeros_like(B3f)

t_start = time.time()

MAX_TIME = 100

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
	FCm_output_pre = np.dot(FCm, inputs) + Bm
	FCi_output_pre = np.dot(FCi, inputs) + Bi
	FCo_output_pre = np.dot(FCo, inputs) + Bo
	FCf_output_pre = np.dot(FCf, inputs) + Bf
	
	FCf_output = 1 / (1 + np.exp(-FCf_output_pre))
	FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
	FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
	#FCm_output = 1 / (1 + np.exp(-FCm_output_pre)) - .5
	FCm_output = FCm_output_pre
	#FCi_output = FCi_output_pre
	#FCf_output = FCf_output_pre
	#FCo_output = FCo_output_pre
	
	FC_output = FCo_output * (FCf_output * CEC + FCi_output * FCm_output)
	
	CEC_kept = CEC*FCf_output
	CEC_new = FCi_output*FCm_output
	
	CEC = CEC*FCf_output + FCi_output*FCm_output
	
	FC2f_output_pre = np.dot(FC2f, FC_output) + B2f
	FC2o_output_pre = np.dot(FC2o, FC_output) + B2o
	FC2i_output_pre = np.dot(FC2i, FC_output) + B2i
	FC2m_output_pre = np.dot(FC2m, FC_output) + B2m
	
	FC2f_output = 1 / (1 + np.exp(-FC2f_output_pre))
	FC2o_output = 1 / (1 + np.exp(-FC2o_output_pre))
	FC2i_output = 1 / (1 + np.exp(-FC2i_output_pre))
	#FC2m_output = 1 / (1 + np.exp(-FC2m_output_pre)) - .5
	FC2m_output = FC2m_output_pre
	#FC2i_output = FC2i_output_pre
	#FC2o_output = FC2o_output_pre
	#FC2f_output = FC2f_output_pre
	
	FC2_output = FC2o_output * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	CEC2_kept = CEC2*FC2f_output
	CEC2_new = FC2i_output*FC2m_output
	
	CEC2 = CEC2*FC2f_output + FC2i_output*FC2m_output
	
	FC3f_output_pre = np.dot(FC3f, FC2_output) + B3f
	FC3o_output_pre = np.dot(FC3o, FC2_output) + B3o
	FC3i_output_pre = np.dot(FC3i, FC2_output) + B3i
	FC3m_output_pre = np.dot(FC3m, FC2_output) + B3m
	
	FC3f_output = 1 / (1 + np.exp(-FC3f_output_pre))
	FC3o_output = 1 / (1 + np.exp(-FC3o_output_pre))
	FC3i_output = 1 / (1 + np.exp(-FC3i_output_pre))
	#FC3m_output = 1 / (1 + np.exp(-FC3m_output_pre)) - .5
	FC3m_output = FC3m_output_pre
	#FC3f_output = FC3f_output_pre
	#FC3o_output = FC3o_output_pre
	#FC3i_output = FC3i_output_pre
	
	FC3_output = FC3o_output * (FC3f_output * CEC3 + FC3i_output * FC3m_output)
	
	CEC3_kept = CEC3*FC3f_output
	CEC3_new = FC3i_output*FC3m_output
	
	CEC3 = CEC3*FC3f_output + FC3i_output*FC3m_output
	
	pred = np.dot(FL, FC3_output)
	
	pred_m_Y = pred - target
	
	err_t += pred_m_Y**2
	
	############### reverse pointwise
	FC3f_output_rev = np.exp(FC3f_output_pre)/((np.exp(FC3f_output_pre) + 1)**2)
	FC3o_output_rev = np.exp(FC3o_output_pre)/((np.exp(FC3o_output_pre) + 1)**2)
	FC3i_output_rev = np.exp(FC3i_output_pre)/((np.exp(FC3i_output_pre) + 1)**2)
	#FC3m_output_rev = np.exp(FC3m_output_pre)/((np.exp(FC3m_output_pre) + 1)**2)
	FC3m_output_rev = 1
	#FC3i_output_rev = 1
	#FC3o_output_rev = 1
	#FC3f_output_rev = 1
	
	FC2f_output_rev = np.exp(FC2f_output_pre)/((np.exp(FC2f_output_pre) + 1)**2)
	FC2o_output_rev = np.exp(FC2o_output_pre)/((np.exp(FC2o_output_pre) + 1)**2)
	FC2i_output_rev = np.exp(FC2i_output_pre)/((np.exp(FC2i_output_pre) + 1)**2)
	#FC2m_output_rev = np.exp(FC2m_output_pre)/((np.exp(FC2m_output_pre) + 1)**2)
	FC2m_output_rev = 1
	#FC2f_output_rev = 1
	#FC2o_output_rev = 1
	#FC2i_output_rev = 1
	
	FCf_output_rev = np.exp(FCf_output_pre)/((np.exp(FCf_output_pre) + 1)**2)
	FCi_output_rev = np.exp(FCi_output_pre)/((np.exp(FCi_output_pre) + 1)**2)
	FCo_output_rev = np.exp(FCo_output_pre)/((np.exp(FCo_output_pre) + 1)**2)
	#FCm_output_rev = np.exp(FCm_output_pre)/((np.exp(FCm_output_pre) + 1)**2)
	FCm_output_rev = 1
	#FCo_output_rev = 1
	#FCi_output_rev = 1
	#FCf_output_rev = 1
	
	
	############ FL
	
	dFL = pred_m_Y * FC2_output
	
	above_w = pred_m_Y * FL
	
	######################### mem 3 gradients:
	
	FC3f_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3f * FC3f_output + FC3f_output_rev * CEC3)
	FC3i_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3i * FC3f_output + FC3i_output_rev * FC3m_output)
	FC3m_output_rev_sig = above_w * FC3o_output * (CEC3_dFC3m * FC3f_output + FC3i_output * FC3m_output_rev)
	FC3o_output_rev_sig = above_w * FC3o_output_rev * (FC3f_output * CEC3 + FC3i_output * FC3m_output)
	
	CEC3_dFC3f = CEC3_dFC3f * FC3f_output + FC3f_output_rev * CEC3
	CEC3_dFC3m = CEC3_dFC3m * FC3f_output + FC3i_output * FC3m_output_rev
	CEC3_dFC3i = CEC3_dFC3i * FC3f_output + FC3i_output_rev * FC3m_output
	
	dB3f += FC3f_output_rev_sig
	dB3i += FC3i_output_rev_sig
	dB3m += FC3m_output_rev_sig
	dB3o += FC3o_output_rev_sig
	
	dFC3f += np.einsum(FC_output, [0], FC3f_output_rev_sig, [1], [1,0])
	dFC3i += np.einsum(FC_output, [0], FC3i_output_rev_sig, [1], [1,0])
	dFC3m += np.einsum(FC_output, [0], FC3m_output_rev_sig, [1], [1,0])
	dFC3o += np.einsum(FC_output, [0], FC3o_output_rev_sig, [1], [1,0])
	
	above_w = np.einsum(FC3o, [0,1], FC3o_output_rev_sig, [0], [1])
	above_w += np.einsum(FC3f, [0,1], FC3f_output_rev_sig, [0], [1])
	above_w += np.einsum(FC3i, [0,1], FC3i_output_rev_sig, [0], [1])
	above_w += np.einsum(FC3m, [0,1], FC3m_output_rev_sig, [0], [1])
	
	######################### mem 2 gradients:
	
	FC2f_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2f * FC2f_output + FC2f_output_rev * CEC2)
	FC2i_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2i * FC2f_output + FC2i_output_rev * FC2m_output)
	FC2m_output_rev_sig = above_w * FC2o_output * (CEC2_dFC2m * FC2f_output + FC2i_output * FC2m_output_rev)
	FC2o_output_rev_sig = above_w * FC2o_output_rev * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	CEC2_dFC2f = CEC2_dFC2f * FC2f_output + FC2f_output_rev * CEC2
	CEC2_dFC2m = CEC2_dFC2m * FC2f_output + FC2i_output * FC2m_output_rev
	CEC2_dFC2i = CEC2_dFC2i * FC2f_output + FC2i_output_rev * FC2m_output
	
	dB2f += FC2f_output_rev_sig
	dB2i += FC2i_output_rev_sig
	dB2m += FC2m_output_rev_sig
	dB2o += FC2o_output_rev_sig
	
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
	
	dBf += FCf_output_rev_sig
	dBi += FCi_output_rev_sig
	dBm += FCm_output_rev_sig
	dBo += FCo_output_rev_sig
	
	dFCf += np.einsum(inputs, [0], FCf_output_rev_sig, [1], [1,0])
	dFCi += np.einsum(inputs, [0], FCi_output_rev_sig, [1], [1,0])
	dFCm += np.einsum(inputs, [0], FCm_output_rev_sig, [1], [1,0])
	dFCo += np.einsum(inputs, [0], FCo_output_rev_sig, [1], [1,0])
	
	if global_step % BATCH_SZ == 0:
		#FCf -= dFCf*EPS/BATCH_SZ
		#FCo -= dFCo*EPS/BATCH_SZ
		#FCm -= dFCm*EPS/BATCH_SZ
		#FCi -= dFCi*EPS/BATCH_SZ
		
		Bf -= dBf*EPS/BATCH_SZ
		Bo -= dBo*EPS/BATCH_SZ
		Bm -= dBm*EPS/BATCH_SZ
		Bi -= dBi*EPS/BATCH_SZ
		
		#FC2f -= dFC2f*EPS/BATCH_SZ
		#FC2o -= dFC2o*EPS/BATCH_SZ
		#FC2m -= dFC2m*EPS/BATCH_SZ
		#FC2i -= dFC2i*EPS/BATCH_SZ
		
		B2f -= dB2f*EPS/BATCH_SZ
		B2o -= dB2o*EPS/BATCH_SZ
		B2m -= dB2m*EPS/BATCH_SZ
		B2i -= dB2i*EPS/BATCH_SZ
		
		#FC3f -= dFC3f*EPS/BATCH_SZ
		#FC3o -= dFC3o*EPS/BATCH_SZ
		#FC3m -= dFC3m*EPS/BATCH_SZ
		#FC3i -= dFC3i*EPS/BATCH_SZ
		
		B3f -= dB3f*EPS/BATCH_SZ
		B3o -= dB3o*EPS/BATCH_SZ
		B3m -= dB3m*EPS/BATCH_SZ
		B3i -= dB3i*EPS/BATCH_SZ
		
		#FL -= dFL*EPS/BATCH_SZ
		
		dFCf = np.zeros_like(FCf)
		dFCo = np.zeros_like(FCo)
		dFCm = np.zeros_like(FCm)
		dFCi = np.zeros_like(FCi)
		
		dBf = np.zeros_like(Bf)
		dBo = np.zeros_like(Bo)
		dBm = np.zeros_like(Bm)
		dBi = np.zeros_like(Bi)
		
		dFC2f = np.zeros_like(FC2f)
		dFC2o = np.zeros_like(FC2o)
		dFC2m = np.zeros_like(FC2m)
		dFC2i = np.zeros_like(FC2i)
		
		dB2f = np.zeros_like(B2f)
		dB2o = np.zeros_like(B2o)
		dB2m = np.zeros_like(B2m)
		dB2i = np.zeros_like(B2i)
		
		dFC3f = np.zeros_like(FC3f)
		dFC3o = np.zeros_like(FC3o)
		dFC3m = np.zeros_like(FC3m)
		dFC3i = np.zeros_like(FC3i)
		
		dB3f = np.zeros_like(B3f)
		dB3o = np.zeros_like(B3o)
		dB3m = np.zeros_like(B3m)
		dB3i = np.zeros_like(B3i)
	
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
		savemat('/home/darren/timer_linear.mat', {'inputs_recent': inputs_recent, 'targets_recent': targets_recent, \
			'preds_recent': preds_recent, 'err': err})
	
		t_start = time.time()
		err_t = 0
	
	global_step += 1
	elapsed_time += 1
		
