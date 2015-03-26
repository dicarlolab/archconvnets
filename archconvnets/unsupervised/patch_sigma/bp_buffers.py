from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu

GPU_FORWARD = 0
GPU_SUP = 1
GPU_UNS = 0

N_TEST_SET = 500*2
N_TRAIN = 300*2
TOP_N = 1

# gpu buffer indices
MAX_OUTPUT1_UNS = 0
DF2_DATA_UNS = 1
CONV_OUTPUT1_UNS = 2
DPOOL1_UNS = 3
F1_IND = 4
IMGS_PAD_UNS = 5
DF1_UNS = 6
F2_IND = 11
D_UNS_UNPOOL2 = 12
F3_IND = 13

MAX_OUTPUT2_UNS = 14
MAX_OUTPUT3_UNS = 15

MAX_OUTPUT1_SUP = 16
MAX_OUTPUT2_SUP = 17
MAX_OUTPUT3_SUP = 18

CONV_OUTPUT1_UNS = 19
CONV_OUTPUT2_UNS = 20
CONV_OUTPUT3_UNS = 21

CONV_OUTPUT1_SUP = 22
CONV_OUTPUT2_SUP = 23
CONV_OUTPUT3_SUP = 24
DF2_UNS = 25
DPOOL2_UNS = 26
DF3_DATA_UNS = 27
DPOOL3_UNS = 28
DF3_UNS = 29
FL_PRED_UNS = 30
FL_PRED_SUP = 31
DF3_SUP=32
DPOOL3_SUP = 33
DF3_DATA_SUP=34
DPOOL2_SUP = 35
DF2_SUP = 36
DF2_DATA_SUP=37
IMGS_PAD_SUP=38
DF3_DATA_SUP=39
DPOOL1_SUP=40
DF1_SUP = 41

#kernprof -l bp.py
#python -m line_profiler bp.py.lprof  > p
#@profile
#def sf():
F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

EPS = 1e-3

N_IMGS = 100 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34 # input image size (px)
PAD = 2

N = 16
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 10 # number of categories

max_output_sz3  = 5

np.random.seed(6166)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

t_start = time.time()

epoch = 0
err = []
class_err = []
mcc = []

while True:
	for batch in range(1,6):
		##################
		# load test imgs into buffers
		z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
		x = z['data'] - imgs_mean
		x = x.reshape((3, 32, 32, 10000))

		labels = np.asarray(z['labels'])

		l = np.zeros((10000, N_C),dtype='int')
		l[np.arange(10000),np.asarray(z['labels']).astype(int)] = 1
		Y_test = np.single(l.T)

		imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, 10000),dtype='single')
		imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
		imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))
		for s in range(100):
			# forward pass
			conv_output1 = conv(F1, imgs_pad[s*N_IMGS:(s+1)*N_IMGS], PAD=2)
			max_output1 = max_pool_cudnn(conv_output1)
			conv_output2 = conv(F2, max_output1, PAD=2)
			max_output2 = max_pool_cudnn(conv_output2)
			conv_output3 = conv(F3, max_output2, PAD=2)
			max_output3 = max_pool_cudnn(conv_output3)
			
			max_output_sz3 = max_output3.shape[2]
			max_output_sz2 = max_output2.shape[2]
			max_output_sz1 = max_output1.shape[2]
			
			conv_output_sz3 = conv_output3.shape[2]
			conv_output_sz2 = conv_output2.shape[2]
			conv_output_sz1 = conv_output1.shape[2]
			
			Ys = Y_test[:,s*N_IMGS:(s+1)*N_IMGS]
			pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4])
			
			
			
			######## gradients:
			
			dFL_uns = np.einsum(max_output3, range(4), pred, [4,0], [4,1,2,3])
			dFL_s = np.einsum(max_output3, range(4), -Ys, [4,0], [4,1,2,3])
			
			grad_FL = 2*(dFL_uns + dFL_s)
			
			###### ravel together categories and imgs, replicate across imgs (FL weights) or categories (switches) as necessary to match dims.
			FL_pred = np.einsum(FL, range(4), pred, [0,4], [0,4,1,2,3]).reshape((N_C*N_IMGS, n3, max_output_sz3, max_output_sz3))
			FL_Y = np.einsum(FL, range(4), Ys, [0,4], [0,4,1,2,3]).reshape((N_C*N_IMGS, n3, max_output_sz3, max_output_sz3))
			
			imgs_pads = np.tile(imgs_pad[s*100:(s+1)*100],(N_C,1,1,1,1)).reshape((N_C*N_IMGS, 3, IMG_SZ, IMG_SZ))
			# each category's predictions are weighted differently, but they all use the same switches
			conv_output3 = np.tile(conv_output3,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n3, conv_output_sz3, conv_output_sz3))
			conv_output2 = np.tile(conv_output2,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n2, conv_output_sz2, conv_output_sz2))
			conv_output1 = np.tile(conv_output1,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n1, conv_output_sz1, conv_output_sz1))
			
			max_output3 = np.tile(max_output3,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n3, max_output_sz3, max_output_sz3))
			max_output2 = np.tile(max_output2,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n2, max_output_sz2, max_output_sz2))
			max_output1 = np.tile(max_output1,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n1, max_output_sz1, max_output_sz1))
			FL_rep_imgs = np.tile(FL,(N_IMGS,1,1,1,1)).transpose((1,0,2,3,4)).reshape((N_C*N_IMGS, n3, max_output_sz3, max_output_sz3))
			
			######### buffers:
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_UNS)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_UNS)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_UNS)
			
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_SUP)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_SUP)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_SUP)
			
			set_buffer(max_output1, MAX_OUTPUT1_UNS, gpu=GPU_UNS)
			set_buffer(max_output2, MAX_OUTPUT2_UNS, gpu=GPU_UNS)
			set_buffer(max_output3, MAX_OUTPUT3_UNS, gpu=GPU_UNS)
			
			set_buffer(max_output1, MAX_OUTPUT1_SUP, gpu=GPU_SUP)
			set_buffer(max_output2, MAX_OUTPUT2_SUP, gpu=GPU_SUP)
			set_buffer(max_output3, MAX_OUTPUT3_SUP, gpu=GPU_SUP)
			
			set_buffer(conv_output1, CONV_OUTPUT1_UNS, gpu=GPU_UNS)
			set_buffer(conv_output2, CONV_OUTPUT2_UNS, gpu=GPU_UNS)
			set_buffer(conv_output3, CONV_OUTPUT3_UNS, gpu=GPU_UNS)
			
			set_buffer(conv_output1, CONV_OUTPUT1_SUP, gpu=GPU_SUP)
			set_buffer(conv_output2, CONV_OUTPUT2_SUP, gpu=GPU_SUP)
			set_buffer(conv_output3, CONV_OUTPUT3_SUP, gpu=GPU_SUP)
			
			set_buffer(imgs_pads, IMGS_PAD_UNS, gpu=GPU_UNS)
			set_buffer(imgs_pads, IMGS_PAD_SUP, gpu=GPU_SUP)
			
			set_buffer(FL_pred, FL_PRED_UNS, gpu=GPU_UNS)
			set_buffer(-FL_Y, FL_PRED_SUP, gpu=GPU_SUP)
			
			###########

			
			max_pool_back_cudnn_buffers(MAX_OUTPUT3_UNS, FL_PRED_UNS, CONV_OUTPUT3_UNS, DPOOL3_UNS, gpu=GPU_UNS)
			max_pool_back_cudnn_buffers(MAX_OUTPUT3_SUP, FL_PRED_SUP, CONV_OUTPUT3_SUP, DPOOL3_SUP, gpu=GPU_SUP)
			
			conv_dfilter_buffers(F3_IND, MAX_OUTPUT2_UNS, DPOOL3_UNS, DF3_UNS, PAD=2, gpu=GPU_UNS)
			conv_dfilter_buffers(F3_IND, MAX_OUTPUT2_SUP, DPOOL3_SUP, DF3_SUP, PAD=2, gpu=GPU_SUP)
			
			conv_ddata_buffers(F3_IND, MAX_OUTPUT2_UNS, DPOOL3_UNS, DF3_DATA_UNS, PAD=2, gpu=GPU_UNS)
			conv_ddata_buffers(F3_IND, MAX_OUTPUT2_SUP, DPOOL3_SUP, DF3_DATA_SUP, PAD=2, gpu=GPU_SUP)
			
			max_pool_back_cudnn_buffers(MAX_OUTPUT2_UNS, DF3_DATA_UNS, CONV_OUTPUT2_UNS, DPOOL2_UNS, gpu=GPU_UNS)
			max_pool_back_cudnn_buffers(MAX_OUTPUT2_SUP, DF3_DATA_SUP, CONV_OUTPUT2_SUP, DPOOL2_SUP, gpu=GPU_SUP)
			
			conv_ddata_buffers(F2_IND, MAX_OUTPUT1_UNS, DPOOL2_UNS, DF2_DATA_UNS, PAD=2, gpu=GPU_UNS)
			conv_ddata_buffers(F2_IND, MAX_OUTPUT1_SUP, DPOOL2_SUP, DF2_DATA_SUP, PAD=2, gpu=GPU_SUP)
			
			conv_dfilter_buffers(F2_IND, MAX_OUTPUT1_UNS, DPOOL2_UNS, DF2_UNS, PAD=2, gpu=GPU_UNS)
			conv_dfilter_buffers(F2_IND, MAX_OUTPUT1_SUP, DPOOL2_SUP, DF2_SUP, PAD=2, gpu=GPU_SUP)
			
			max_pool_back_cudnn_buffers(MAX_OUTPUT1_UNS, DF2_DATA_UNS, CONV_OUTPUT1_UNS, DPOOL1_UNS, gpu=GPU_UNS)
			max_pool_back_cudnn_buffers(MAX_OUTPUT1_SUP, DF2_DATA_SUP, CONV_OUTPUT1_SUP, DPOOL1_SUP, gpu=GPU_SUP)
			
			conv_dfilter_buffers(F1_IND, IMGS_PAD_UNS, DPOOL1_UNS, DF1_UNS, PAD=2, gpu=GPU_UNS)
			conv_dfilter_buffers(F1_IND, IMGS_PAD_SUP, DPOOL1_SUP, DF1_SUP, PAD=2, gpu=GPU_SUP)
			
			###
			
			dF3_uns = return_buffer(DF3_UNS, gpu=GPU_UNS)
			dF3_s = return_buffer(DF3_SUP, gpu=GPU_SUP)
			
			dF2_uns = return_buffer(DF2_UNS, gpu=GPU_UNS)
			dF2_s = return_buffer(DF2_SUP, gpu=GPU_SUP)
			
			dF1_uns = return_buffer(DF1_UNS, gpu=GPU_UNS)
			dF1_s = return_buffer(DF1_SUP, gpu=GPU_SUP)
			
			
			grad_F3 = 2*(dF3_uns + dF3_s)
			grad_F2 = 2*(dF2_uns + dF2_s)
			grad_F1 = 2*(dF1_uns + dF1_s)
			
			F1 -= grad_F1*EPS / N_IMGS
			F2 -= grad_F2*EPS / N_IMGS
			F3 -= grad_F3*EPS / N_IMGS
			FL -= grad_FL*EPS / N_IMGS
			
		
		conv_output1 = conv(F1, imgs_pad, PAD=2)
		max_output1 = max_pool_cudnn(conv_output1)
		conv_output2 = conv(F2, max_output1, PAD=2)
		max_output2 = max_pool_cudnn(conv_output2)
		conv_output3 = conv(F3, max_output2, PAD=2)
		max_output3 = max_pool_cudnn(conv_output3)
		
		pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4])
		
		err.append(np.mean((pred - Y_test)**2))
		class_err.append(1-(pred.argmax(0) == labels).mean())
		
		## mcc
		t_mcc = time.time()
		pred_train = pred[:,N_TRAIN:N_TEST_SET].T
		pred = pred[:,:N_TRAIN].T
		
		test_corrs = np.dot(pred, pred_train.T)
		hit = 0
		for test_img in range(N_TEST_SET-N_TRAIN):
			hit += np.max(labels[N_TRAIN + test_img] == labels[np.argsort(-test_corrs[test_img])[:TOP_N]])
		mcc.append(1-hit/np.single(N_TEST_SET-N_TRAIN))
		
		print epoch, batch, err[-1], class_err[-1], mcc[-1], np.sum(np.abs(F1)), time.time() - t_start, time.time() - t_mcc
		savemat('/home/darren/F1.mat', {'F1':F1, 'epoch':epoch, 'class_err':class_err, 'err':err,'mcc':mcc})
		t_start = time.time()
	epoch += 1
sf()
