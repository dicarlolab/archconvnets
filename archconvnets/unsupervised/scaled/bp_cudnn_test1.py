from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
import numexpr as ne
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import max_pool_locs
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu

#kernprof -l bp_cudnn.py
#python -m line_profiler bp_cudnn.py.lprof  > p
#@profile
#def sf():

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

EPS = 1e-3#1e-2
eps_F1 = EPS
eps_F2 = EPS
eps_F3 = EPS
eps_FL = EPS

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 10 # batch size
N_TEST_IMGS = N_IMGS #N_SIGMA_IMGS #128*2
IMG_SZ_CROP = 28 # input image size (px)
IMG_SZ = 32 # input image size (px)
img_train_offset = 2
PAD = 2

N = 4
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 10 # number of categories

max_output_sz3  = 24

np.random.seed(6166)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

##################
# load test imgs into buffers
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_TEST_IMGS]

labels = np.asarray(z['labels'])[:N_IMGS]

l = np.zeros((N_TEST_IMGS, N_C),dtype='int')
l[np.arange(N_TEST_IMGS),np.asarray(z['labels'])[:N_TEST_IMGS].astype(int)] = 1
Y_test = np.single(l.T)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_TEST_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP]
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

FLr = FL.reshape((N_C, n3*max_output_sz3**2))

EPS = 5e-7
print np.sum(np.abs(F1))
for s in range(10000):
	conv_output1 = conv(F1, imgs_pad)
	conv_output2 = conv(F2, conv_output1)
	pred_unsum = np.einsum(FL, range(4), conv_output2, [4,1,2,3], [0,4,1,2,3])
	pred = np.einsum(pred_unsum, range(5), [0,1])

	pred_uns2 = np.einsum(conv_output2, range(4), FL**2, [4,1,2,3], [4,0,1,2,3])
	
	grad = np.zeros_like(F1)
	for cat_i in range(N_C):
		dconv_output1 = conv_ddata(F2, conv_output1, pred_uns2[cat_i])
		dconv_output1_1 = conv_ddata(F2, conv_output1, Y_test[cat_i,:,np.newaxis,np.newaxis,np.newaxis]*FL[cat_i])
		
		dF1_uns = conv_dfilter(F1, imgs_pad, dconv_output1)
		dF1_uns_1 = conv_dfilter(F1, imgs_pad, dconv_output1_1)
		
		grad += dF1_uns - dF1_uns_1
	F1 -= grad*EPS
	
	print np.sum((pred - Y_test)**2)/N_TEST_IMGS, np.sum(pred**2), np.sum(np.abs(F1)), (pred.argmax(0) == labels).sum()/np.single(N_IMGS)
	if s % 5 == 0:
		print 'saving'
		savemat('/home/darren/F1.mat', {'F1':F1})
