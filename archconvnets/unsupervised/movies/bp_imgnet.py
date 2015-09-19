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
F2_scale = 0.001
F3_scale = 0.001
FL_scale = 0.001

EPS_E = 3
EPS = 5*10**(-EPS_E)

N_IMGS = 100 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34 # input image size (px)
PAD = 2

GPU_S = 2
GPU_UNS = 3

N = 32
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

file_name = '/home/darren/F1_' + str(N_C) + '_' + str(EPS_E) + 'eps_' + str(N) + 'N_imgnet.mat'

max_output_sz3  = 5

np.random.seed(6166)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))
FL_cifar = np.single(np.random.normal(scale=FL_scale, size=(10, n3, max_output_sz3, max_output_sz3)))

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
# load test imgs into buffers (imgnet)
z2 = np.load('/export/storage/imgnet32/data_batch_' + str(1))
x = z2['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))

labels_test_imgnet = np.asarray(z2['labels'])
l = np.zeros((10000, 999),dtype='int')
l[np.arange(10000),np.asarray(z2['labels']).astype(int)] = 1
Y_test_imgnet = np.single(l.T)

imgs_pad_test_imgnet = np.zeros((3, IMG_SZ, IMG_SZ, 10000),dtype='single')
imgs_pad_test_imgnet[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
imgs_pad_test_imgnet = np.ascontiguousarray(imgs_pad_test_imgnet.transpose((3,0,1,2)))

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
err_imgnet = []
class_err_imgnet = []

global_step = 0
while True:
	for batch in range(2,106):
		t_mcc = time.time()
		if batch % 1 == 0:
			###############################################
			# test imgs (cifar)
			conv_output1 = conv(F1, imgs_pad_test, gpu=GPU_UNS)
			max_output1 = max_pool_cudnn(conv_output1, gpu=GPU_UNS)
			conv_output2 = conv(F2, max_output1, gpu=GPU_UNS)
			max_output2 = max_pool_cudnn(conv_output2, gpu=GPU_UNS)
			conv_output3 = conv(F3, max_output2, gpu=GPU_UNS)
			max_output3 = max_pool_cudnn(conv_output3, gpu=GPU_UNS)
			
			pred = np.einsum(FL_cifar, range(4), max_output3, [4,1,2,3], [0,4])
			
			err.append(np.mean((pred - Y_test)**2))
			class_err.append(1-(np.argmax(pred,axis=0) == np.asarray(np.squeeze(labels_test))).mean())
			
			###############################################
			# test imgs (imgnet)
			conv_output1 = conv(F1, imgs_pad_test_imgnet, gpu=GPU_UNS)
			max_output1 = max_pool_cudnn(conv_output1, gpu=GPU_UNS)
			conv_output2 = conv(F2, max_output1, gpu=GPU_UNS)
			max_output2 = max_pool_cudnn(conv_output2, gpu=GPU_UNS)
			conv_output3 = conv(F3, max_output2, gpu=GPU_UNS)
			max_output3 = max_pool_cudnn(conv_output3, gpu=GPU_UNS)
			
			pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4])
			
			err_imgnet.append(np.mean((pred - Y_test_imgnet)**2))
			class_err_imgnet.append(1-(np.argmax(pred,axis=0) == np.asarray(np.squeeze(labels_test_imgnet))).mean())
			
			print epoch, batch, 'class:', class_err[-1], 'err:', err[-1], ' F1:', np.sum(np.abs(F1)), time.time() - t_mcc, time.time() - t_start, file_name
			print '       class imgnet:', class_err_imgnet[-1], 'err imgnet:', err_imgnet[-1]
			savemat(file_name, {'F1':F1, 'epoch':epoch, 'class_err':class_err, 'err':err,
				'class_err_imgnet':class_err_imgnet, 'err':err_imgnet,
				'F2':F2,'F3':F3,'FL':FL,'EPS':EPS,'err':err,'class_err':class_err})
				
			t_start = time.time()
	
		##################
		# load train imgs into buffers
		z = np.load('/export/storage/imgnet32/data_batch_' + str(batch))
		
		imgs_pad = np.zeros((10000, 3, IMG_SZ, IMG_SZ),dtype='single')
		Y_train = np.zeros((N_C, 10000), dtype='uint8')

		x = z['data'] - z['mean']
		x = x.reshape((3, 32, 32, 10000))

		labels = np.asarray(z['labels'])

		l = np.zeros((10000, N_C),dtype='uint8')
		l[np.arange(10000),np.asarray(z['labels']).astype(int)] = 1
		Y_train = l.T

		imgs_pad[:,:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x.transpose((3,0,1,2))
		imgs_pad = np.ascontiguousarray(imgs_pad)
		for s in range(100):
			s_cifar = global_step % 500
			
			set_buffer(imgs_pad[s*N_IMGS:(s+1)*N_IMGS], IMGS_PAD, gpu=GPU_UNS)
			set_buffer(imgs_pad_cifar[s_cifar*N_IMGS:(s_cifar+1)*N_IMGS], IMGS_PAD, gpu=GPU_S)
			
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_UNS)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_UNS)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_UNS)
			set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_UNS)
			
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_S)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_S)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_S)
			set_buffer(FL_cifar, FL_IND, filter_flag=1, gpu=GPU_S)
			
			# forward pass imgs
			conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_UNS)
			max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_UNS)
			conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_UNS)
			max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_UNS)
			conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_UNS)
			max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_UNS)
			
			# forward pass imgs (cifar)
			conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_S)
			max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_S)
			conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_S)
			max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_S)
			conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_S)
			max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_S)
			
			Ys = np.ascontiguousarray(Y_train[:,s*N_IMGS:(s+1)*N_IMGS])
			Ys_cifar = np.ascontiguousarray(Y_cifar[:,s_cifar*N_IMGS:(s_cifar+1)*N_IMGS])
			
			######## gradients:
			pred_buffer(FL_IND, MAX_OUTPUT3, PRED, Ys, gpu=GPU_UNS) # === np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4]) - Ys
			pred_buffer(FL_IND, MAX_OUTPUT3, PRED, Ys_cifar, gpu=GPU_S) # === np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4]) - Ys
			
			pred = return_2d_buffer(PRED, gpu=GPU_UNS)
			pred_cifar = return_2d_buffer(PRED, gpu=GPU_S)
			
			FL_pred = np.einsum(FL, range(4), pred, [0,4], [4,1,2,3])
			
			set_buffer(FL_pred, FL_PRED, gpu=GPU_UNS) # summing across categories
			
			###########
			max_pool_back_cudnn_buffers(MAX_OUTPUT3, FL_PRED, CONV_OUTPUT3, DPOOL3, gpu=GPU_UNS)
			conv_dfilter_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3, stream=3, gpu=GPU_UNS)
			conv_ddata_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3_DATA, gpu=GPU_UNS)
			max_pool_back_cudnn_buffers(MAX_OUTPUT2, DF3_DATA, CONV_OUTPUT2, DPOOL2, gpu=GPU_UNS)
			conv_ddata_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2_DATA, gpu=GPU_UNS)
			conv_dfilter_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2, stream=2, gpu=GPU_UNS)
			max_pool_back_cudnn_buffers(MAX_OUTPUT1, DF2_DATA, CONV_OUTPUT1, DPOOL1, gpu=GPU_UNS)
			conv_dfilter_buffers(F1_IND, IMGS_PAD, DPOOL1, DF1, stream=1, gpu=GPU_UNS)

			###
			max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_UNS)
			max_output3_cifar = return_buffer(MAX_OUTPUT3, gpu=GPU_S)
			
			dFL = np.einsum(max_output3, range(4), pred - Ys, [4,0], [4,1,2,3]) 
			dFL_cifar = np.einsum(max_output3_cifar, range(4), pred_cifar - Ys_cifar, [4,0], [4,1,2,3]) 
			
			dF3 = return_buffer(DF3, stream=3, gpu=GPU_UNS)
			dF2 = return_buffer(DF2, stream=2, gpu=GPU_UNS)
			dF1 = return_buffer(DF1, stream=1, gpu=GPU_UNS)
			
			
			F1 -= dF1*EPS / N_IMGS
			F2 -= dF2*EPS / N_IMGS
			F3 -= dF3*EPS / N_IMGS
			FL -= dFL*EPS / N_IMGS
			FL_cifar -= dFL_cifar*EPS / N_IMGS
			
			global_step += 1
		
	epoch += 1
sf()
