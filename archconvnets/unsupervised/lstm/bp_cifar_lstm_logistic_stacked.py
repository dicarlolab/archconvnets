from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu

#kernprof -l bp_movies_nat.py
#python -m line_profiler bp_movies_nat.py.lprof  > p
#@profile
#def sf():

N_C = 999

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = .1
FL2_scale = .1
CEC_SCALE = 0.01

EPS_E = 2
EPS = 1*10**(-EPS_E)

N_IMGS = 100 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34 # input image size (px)
PAD = 2

GPU_UNS = 1

N = 32
n1 = N # L1 filters
n2 = N# ...
n3 = N
n4 = N+1
n5 = N+2

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

file_name = '/home/darren/F1_' + str(N_C) + '_' + str(EPS_E) + 'eps_' + str(N) + 'N_cifar.mat'

max_output_sz3  = 5

np.random.seed(6166)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))

FCm = np.single(np.random.normal(scale=FL_scale, size=(n4, n3, max_output_sz3, max_output_sz3)))
FCi = np.single(np.random.normal(scale=FL_scale, size=(n4, n3, max_output_sz3, max_output_sz3)))
FCo = np.single(np.random.normal(scale=FL_scale, size=(n4, n3, max_output_sz3, max_output_sz3)))
FCf = np.single(np.random.normal(scale=FL_scale, size=(n4, n3, max_output_sz3, max_output_sz3)))
CEC = np.single(np.random.normal(scale=CEC_SCALE, size=(N_IMGS, n4)))
CECt = np.single(np.random.normal(scale=CEC_SCALE, size=(10000, n4)))

FC2f = np.single(np.random.normal(scale=FL2_scale, size=(n5, n4)))
FC2o = np.single(np.random.normal(scale=FL2_scale, size=(n5, n4)))
FC2i = np.single(np.random.normal(scale=FL2_scale, size=(n5, n4)))
FC2m = np.single(np.random.normal(scale=FL2_scale, size=(n5, n4)))

CEC2 = np.single(np.random.normal(scale=CEC_SCALE, size=(N_IMGS, n5)))
CEC2t = np.single(np.random.normal(scale=CEC_SCALE, size=(10000, n5)))

FL = np.single(np.random.normal(scale=1, size=(10, n5)))

#CEC = np.zeros_like(CEC)
#CEC2 = np.zeros_like(CEC2)

CEC_kept = 0; CEC_new = 0
CEC2_kept = 0; CEC2_new = 0
dF1 = np.zeros_like(F1)
dF2 = np.zeros_like(F2)
dF3 = np.zeros_like(F3)
dFL = np.zeros_like(FL)
dFCm = np.zeros_like(FCm)
dFCi = np.zeros_like(FCi)
dFCo = np.zeros_like(FCo)
dFCf = np.zeros_like(FCf)
dFC2m = np.zeros_like(FC2m)
dFC2i = np.zeros_like(FC2i)
dFC2o = np.zeros_like(FC2o)
dFC2f = np.zeros_like(FC2f)


imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']


# gpu buffer indices
MAX_OUTPUT1 = 0; DF2_DATA = 1; CONV_OUTPUT1 = 2; DPOOL1 = 3
F1_IND = 4; IMGS_PAD = 5; DF1 = 6; F2_IND = 11
D_UNPOOL2 = 12;F3_IND = 13; MAX_OUTPUT2 = 14; MAX_OUTPUT3 = 15
CONV_OUTPUT1 = 19; CONV_OUTPUT2 = 20; CONV_OUTPUT3 = 21
DF2 = 25; DPOOL2 = 26; DF3_DATA = 27; DPOOL3 = 28; DF3 = 29; FL_PRED = 30;
FL_IND = 31; PRED = 32; DFL = 33

t_start = time.time()

##################
# load test imgs into buffers
z2 = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(6))
x = z2['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))

labels_test = np.asarray(z2['labels'])
l = np.zeros((10000, 10),dtype='int')
l[np.arange(10000),np.asarray(z2['labels']).astype(int)] = 1
Y_test = np.single(l.T)

imgs_pad_test = np.zeros((3, IMG_SZ, IMG_SZ, 10000),dtype='single')
imgs_pad_test[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
imgs_pad_test = np.ascontiguousarray(imgs_pad_test.transpose((3,0,1,2)))

##################
# load cifar train imgs into buffers
z2 = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(1))
for batch in range(2,6):
	y = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
	z2['data'] = np.concatenate((z2['data'], y['data']), axis=1)
	z2['labels'] = np.concatenate((z2['labels'], y['labels']))
	
x = z2['data'] - imgs_mean
x = x.reshape((3, 32, 32, 50000))

labels_cifar = np.asarray(z2['labels'])
l = np.zeros((50000, 10),dtype='uint8')
l[np.arange(50000),np.asarray(z2['labels']).astype(int)] = 1
Y_cifar = l.T

imgs_pad_cifar = np.zeros((3, IMG_SZ, IMG_SZ, 50000),dtype='single')
imgs_pad_cifar[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
imgs_pad_cifar = np.ascontiguousarray(imgs_pad_cifar.transpose((3,0,1,2)))

epoch = 0
err = []
class_err = []

global_step = 0
while True:
	for s in range(500):
		if s % 100 == 0:
			t_mcc = time.time()
			###############################################
			# test imgs (cifar)
			conv_output1 = conv(F1, imgs_pad_test, gpu=GPU_UNS)
			max_output1 = max_pool_cudnn(conv_output1, gpu=GPU_UNS)
			conv_output2 = conv(F2, max_output1, gpu=GPU_UNS)
			max_output2 = max_pool_cudnn(conv_output2, gpu=GPU_UNS)
			conv_output3 = conv(F3, max_output2, gpu=GPU_UNS)
			max_output3 = max_pool_cudnn(conv_output3, gpu=GPU_UNS)
			
			FCm_output_pre = np.einsum(FCm, range(4), max_output3, [4, 1,2,3], [4, 0])
			FCi_output_pre = np.einsum(FCi, range(4), max_output3, [4, 1,2,3], [4, 0])
			FCo_output_pre = np.einsum(FCo, range(4), max_output3, [4, 1,2,3], [4, 0])
			FCf_output_pre = np.einsum(FCf, range(4), max_output3, [4, 1,2,3], [4, 0])
			
			FCf_output = 1 / (1 + np.exp(-FCf_output_pre))
			FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
			FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
			FCm_output = FCm_output_pre
			
			FC_output = FCo_output * (FCf_output * CECt + FCi_output * FCm_output)
			
			CECt = CECt*FCf_output + FCi_output*FCm_output
			
			FC2f_output_pre = np.einsum(FC2f, [0,1], FC_output, [2,1], [2,0])
			FC2o_output_pre = np.einsum(FC2o, [0,1], FC_output, [2,1], [2,0])
			FC2i_output_pre = np.einsum(FC2i, [0,1], FC_output, [2,1], [2,0])
			FC2m_output_pre = np.einsum(FC2m, [0,1], FC_output, [2,1], [2,0])
			
			FC2f_output = 1 / (1 + np.exp(-FC2f_output_pre))
			FC2o_output = 1 / (1 + np.exp(-FC2o_output_pre))
			FC2i_output = 1 / (1 + np.exp(-FC2i_output_pre))
			FC2m_output = FC2m_output_pre
			
			FC2_output = FC2o_output * (FC2f_output * CEC2t + FC2i_output * FC2m_output)
			
			CEC2t = CEC2t*FC2f_output + FC2i_output*FC2m_output
			
			pred = np.einsum(FL, [0,1], FC2_output, [2, 1], [2,0])
			
			err.append(np.mean((pred - Y_test.T)**2))
			class_err.append(1-(np.argmax(pred.T,axis=0) == np.asarray(np.squeeze(labels_test))).mean())
			
			print '---------------------------------------------'
			print epoch, batch, 'class:', class_err[-1], 'err:', err[-1], ' F1:', np.sum(np.abs(F1)), time.time() - t_mcc, time.time() - t_start, file_name
			ft = EPS/N_IMGS
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
			savemat(file_name, {'F1':F1, 'epoch':epoch, 'class_err':class_err, 'err':err,'F2':F2,'F3':F3,'EPS':EPS})
			
			t_start = time.time()
		
		set_buffer(imgs_pad_cifar[s*N_IMGS:(s+1)*N_IMGS], IMGS_PAD, gpu=GPU_UNS)
		
		set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_UNS)
		set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_UNS)
		set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_UNS)
		
		# forward pass imgs
		conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_UNS)
		max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_UNS)
		conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_UNS)
		max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_UNS)
		conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_UNS)
		max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_UNS)
		
		max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_UNS)
		FCm_output_pre = np.einsum(FCm, range(4), max_output3, [4, 1,2,3], [4, 0])
		FCi_output_pre = np.einsum(FCi, range(4), max_output3, [4, 1,2,3], [4, 0])
		FCo_output_pre = np.einsum(FCo, range(4), max_output3, [4, 1,2,3], [4, 0])
		FCf_output_pre = np.einsum(FCf, range(4), max_output3, [4, 1,2,3], [4, 0])
		
		FCf_output = 1 / (1 + np.exp(-FCf_output_pre))
		FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
		FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
		FCm_output = FCm_output_pre
		
		FC_output = FCo_output * (FCf_output * CEC + FCi_output * FCm_output)
		
		CEC_kept = CEC*FCf_output
		CEC_new = FCi_output*FCm_output
		
		CEC = CEC*FCf_output + FCi_output*FCm_output
		
		FC2f_output_pre = np.einsum(FC2f, [0,1], FC_output, [2,1], [2,0])
		FC2o_output_pre = np.einsum(FC2o, [0,1], FC_output, [2,1], [2,0])
		FC2i_output_pre = np.einsum(FC2i, [0,1], FC_output, [2,1], [2,0])
		FC2m_output_pre = np.einsum(FC2m, [0,1], FC_output, [2,1], [2,0])
		
		FC2f_output = 1 / (1 + np.exp(-FC2f_output_pre))
		FC2o_output = 1 / (1 + np.exp(-FC2o_output_pre))
		FC2i_output = 1 / (1 + np.exp(-FC2i_output_pre))
		FC2m_output = FC2m_output_pre
		
		FC2_output = FC2o_output * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
		
		CEC2_kept = CEC2*FC2f_output
		CEC2_new = FC2i_output*FC2m_output
		
		CEC2 = CEC2*FC2f_output + FC2i_output*FC2m_output
		
		pred = np.einsum(FL, [0,1], FC2_output, [2, 1], [2,0])
		
		Ys = np.ascontiguousarray(Y_cifar[:,s*N_IMGS:(s+1)*N_IMGS])
		pred_m_Y = pred - Ys.T
		
		############### reverse pointwise
	
		FC2f_output_rev = np.exp(FC2f_output_pre)/((np.exp(FC2f_output_pre) + 1)**2)
		FC2o_output_rev = np.exp(FC2o_output_pre)/((np.exp(FC2o_output_pre) + 1)**2)
		FC2i_output_rev = np.exp(FC2i_output_pre)/((np.exp(FC2i_output_pre) + 1)**2)
		FC2m_output_rev = 1
		
		FCf_output_rev = np.exp(FCf_output_pre)/((np.exp(FCf_output_pre) + 1)**2)
		FCi_output_rev = np.exp(FCi_output_pre)/((np.exp(FCi_output_pre) + 1)**2)
		FCo_output_rev = np.exp(FCo_output_pre)/((np.exp(FCo_output_pre) + 1)**2)
		FCm_output_rev = 1
		
		
		############ FL
		
		dFL = np.einsum(pred_m_Y, [0,1], FC2_output, [0,2], [1,2])
		
		above_w = np.dot(pred_m_Y, FL)
		
		######################### mem 2 gradients:
		
		FC2f_output_rev_sig = above_w * FC2o_output * (FC2f_output_rev * CEC2)
		FC2i_output_rev_sig = above_w * FC2o_output * (FC2i_output_rev * FC2m_output)
		FC2m_output_rev_sig = above_w * FC2o_output * (FC2i_output * FC2m_output_rev)
		FC2o_output_rev_sig = above_w * FC2o_output_rev * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
		
		dFC2f = np.einsum(FC_output, [0,1], FC2f_output_rev_sig, [0,2], [2,1])*1e3
		dFC2i = np.einsum(FC_output, [0,1], FC2i_output_rev_sig, [0,2], [2,1])*1e3
		dFC2m = np.einsum(FC_output, [0,1], FC2m_output_rev_sig, [0,2], [2,1])
		dFC2o = np.einsum(FC_output, [0,1], FC2o_output_rev_sig, [0,2], [2,1])*1e3
		
		above_w = np.einsum(FC2o, [0,1], FC2o_output_rev_sig, [2,0], [2,1])
		above_w += np.einsum(FC2f, [0,1], FC2f_output_rev_sig, [2,0], [2,1])
		above_w += np.einsum(FC2i, [0,1], FC2i_output_rev_sig, [2,0], [2,1])
		above_w += np.einsum(FC2m, [0,1], FC2m_output_rev_sig, [2,0], [2,1])
		
		########################## mem 1 gradients:
	
		FCf_output_rev_sig = above_w * FCo_output * (FCf_output_rev * CEC)
		FCi_output_rev_sig = above_w * FCo_output * (FCi_output_rev * FCm_output)
		FCm_output_rev_sig = above_w * FCo_output * (FCi_output * FCm_output_rev)
		FCo_output_rev_sig = above_w * FCo_output_rev * (FCf_output * CEC + FCi_output * FCm_output)
		
		dFCf = np.einsum(max_output3, range(4), FCf_output_rev_sig, [0,4], [4,1,2,3])*1e3
		dFCi = np.einsum(max_output3, range(4), FCi_output_rev_sig, [0,4], [4,1,2,3])*1e3
		dFCm = np.einsum(max_output3, range(4), FCm_output_rev_sig, [0,4], [4,1,2,3])
		dFCo = np.einsum(max_output3, range(4), FCo_output_rev_sig, [0,4], [4,1,2,3])*1e3
		
		above_w = np.einsum(FCo, range(4), FCo_output_rev_sig, [4,0], [4,1,2,3])
		above_w += np.einsum(FCi, range(4), FCi_output_rev_sig, [4,0], [4,1,2,3])
		above_w += np.einsum(FCm, range(4), FCm_output_rev_sig, [4,0], [4,1,2,3])
		above_w += np.einsum(FCf, range(4), FCf_output_rev_sig, [4,0], [4,1,2,3])
		
		set_buffer(above_w, FL_PRED, gpu=GPU_UNS) # summing across categories
		
		###########
		max_pool_back_cudnn_buffers(MAX_OUTPUT3, FL_PRED, CONV_OUTPUT3, DPOOL3, gpu=GPU_UNS)
		conv_dfilter_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3, stream=3, gpu=GPU_UNS)
		conv_ddata_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3_DATA, gpu=GPU_UNS)
		max_pool_back_cudnn_buffers(MAX_OUTPUT2, DF3_DATA, CONV_OUTPUT2, DPOOL2, gpu=GPU_UNS)
		conv_ddata_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2_DATA, gpu=GPU_UNS)
		conv_dfilter_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2, stream=2, gpu=GPU_UNS)
		max_pool_back_cudnn_buffers(MAX_OUTPUT1, DF2_DATA, CONV_OUTPUT1, DPOOL1, gpu=GPU_UNS)
		conv_dfilter_buffers(F1_IND, IMGS_PAD, DPOOL1, DF1, stream=1, gpu=GPU_UNS)


		dF3 = return_buffer(DF3, stream=3, gpu=GPU_UNS)
		dF2 = return_buffer(DF2, stream=2, gpu=GPU_UNS)
		dF1 = return_buffer(DF1, stream=1, gpu=GPU_UNS)
		
		F1 -= dF1*EPS / N_IMGS
		F2 -= dF2*EPS / N_IMGS
		F3 -= dF3*EPS / N_IMGS
		
		FCf -= dFCf*EPS / N_IMGS
		FCo -= dFCo*EPS / N_IMGS
		FCm -= dFCm*EPS / N_IMGS
		FCi -= dFCi*EPS / N_IMGS
		
		FC2f -= dFC2f*EPS / N_IMGS
		FC2o -= dFC2o*EPS / N_IMGS
		FC2m -= dFC2m*EPS / N_IMGS
		FC2i -= dFC2i*EPS / N_IMGS
		
		FL -= dFL*EPS / N_IMGS
		
		global_step += 1
		
	epoch += 1
sf()
