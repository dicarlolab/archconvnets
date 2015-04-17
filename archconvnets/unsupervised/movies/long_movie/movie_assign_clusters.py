from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import pickle as pk
import os

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

EPS_E = 3
EPS = 2*10**(-EPS_E)

N_IMGS = 100 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34 # input image size (px)
PAD = 2

N_C = 101 # number of categories
BP_STR = ''
GPU_S = 3
GPU_S2 = 0
GPU_UNS = 2
s_scale = 1

N = 32
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

max_output_sz3  = 5

np.random.seed(6166)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))
FL_cifar = np.single(np.random.normal(scale=FL_scale, size=(10, n3, max_output_sz3, max_output_sz3)))
FL_imgnet = np.single(np.random.normal(scale=FL_scale, size=(999, n3, max_output_sz3, max_output_sz3)))

imgs_mean = np.load('/home/darren/long_movie_batch/1_movie_batch')['data'].mean(1)[:,np.newaxis]

##################
# load train imgs into buffers
imgs_pad = np.zeros((9600, 3, IMG_SZ, IMG_SZ),dtype='single')

z = np.load('/home/darren/long_movie_batch/1_movie_batch')

z['labels'] = z['labels_obj']

x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 9600))

labels = np.asarray(z['labels'])

imgs_pad[:,:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x.transpose((3,0,1,2))
imgs_pad = np.ascontiguousarray(imgs_pad)

conv_output1 = conv(F1, imgs_pad, gpu=GPU_UNS)
max_output1 = max_pool_cudnn(conv_output1, gpu=GPU_UNS)
conv_output2 = conv(F2, max_output1, gpu=GPU_UNS)
max_output2 = max_pool_cudnn(conv_output2, gpu=GPU_UNS)
conv_output3 = conv(F3, max_output2, gpu=GPU_UNS)
max_output3 = max_pool_cudnn(conv_output3, gpu=GPU_UNS)

max_output3 = zscore(max_output3.reshape((9600,32*5*5)),axis=1)


obj_order = np.zeros(8*8*100*150, dtype='int')

for batch in range(5,97):
	t_start = time.time()
	##################
	# load train imgs into buffers
	imgs_pad_test = np.zeros((10000, 3, IMG_SZ, IMG_SZ),dtype='single')

	z = np.load('/home/darren/long_movie_batch/data_batch_longer_cont_' + str(batch))

	z['labels'] = z['labels_obj']

	x = z['data'] - imgs_mean
	x = x.reshape((3, 32, 32, 10000))

	labels_test = np.asarray(z['labels'])

	imgs_pad_test[:,:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x.transpose((3,0,1,2))
	imgs_pad_test = np.ascontiguousarray(imgs_pad_test)

	conv_output1 = conv(F1, imgs_pad_test, gpu=GPU_UNS)
	max_output1 = max_pool_cudnn(conv_output1, gpu=GPU_UNS)
	conv_output2 = conv(F2, max_output1, gpu=GPU_UNS)
	max_output2 = max_pool_cudnn(conv_output2, gpu=GPU_UNS)
	conv_output3 = conv(F3, max_output2, gpu=GPU_UNS)
	max_output3_test = max_pool_cudnn(conv_output3, gpu=GPU_UNS)
	
	max_output3_test = zscore(max_output3_test.reshape((10000,32*5*5)),axis=1)
	
	corrs = np.einsum(max_output3_test, [0,2], max_output3, [1,2], [0,1])
	print np.mean(labels[corrs.argmax(1)] == labels_test), 1.0/64
	
	label_prev = labels_test[0]
	pred = np.zeros(64)
	movie_start = 0
	hit = 0
	n_movies = 0
	for frame in range(10000):
		if label_prev != labels_test[frame]: # transition
			corrs_movie = corrs[movie_start:frame-1]
			
			frame_preds = labels[corrs_movie.argmax(1)]
			label_est = np.argmax(np.bincount(frame_preds))
			obj_order[(frame + (batch-1)*10000 - 1) / 150] = label_est
			
			if label_est == labels_test[movie_start]:
				hit += 1
			
			label_prev = labels_test[frame]
			movie_start = frame
			n_movies += 1

	print batch, hit/np.single(n_movies), 1.0/64, n_movies
	
	file = open('/home/darren/long_movie_batch/labels_pred', 'w')
	pk.dump({'obj_order': obj_order}, file)
	file.close()
	
	print time.time() - t_start