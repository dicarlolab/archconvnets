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

N = 4
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
			
			
			d_uns = max_pool_back_cudnn(max_output3, FL_pred, conv_output3, gpu=GPU_UNS)
			d_s = max_pool_back_cudnn(max_output3, -FL_Y, conv_output3, gpu=GPU_SUP)

			dF3_uns = conv_dfilter(F3, max_output2, d_uns, PAD=2, gpu=GPU_UNS)
			dF3_s = conv_dfilter(F3, max_output2, d_s, PAD=2, gpu=GPU_SUP)
			
			d_uns = conv_ddata(F3, max_output2, d_uns, PAD=2, gpu=GPU_UNS)
			d_s = conv_ddata(F3, max_output2, d_s, PAD=2, gpu=GPU_SUP)
			
			d_uns = max_pool_back_cudnn(max_output2, d_uns, conv_output2, gpu=GPU_UNS)
			d_s = max_pool_back_cudnn(max_output2, d_s, conv_output2, gpu=GPU_SUP)
			
			d_uns = conv_ddata(F2, max_output1, d_uns, PAD=2, gpu=GPU_UNS)
			d_s = conv_ddata(F2, max_output1, d_s, PAD=2, gpu=GPU_SUP)
			
			dF2_uns = conv_dfilter(F2, max_output1, d_uns, PAD=2, gpu=GPU_UNS)
			dF2_s = conv_dfilter(F2, max_output1, d_s, PAD=2, gpu=GPU_SUP)
			
			d_uns = max_pool_back_cudnn(max_output1, d_uns, conv_output1, gpu=GPU_UNS)
			d_s = max_pool_back_cudnn(max_output1, d_s, conv_output1, gpu=GPU_SUP)
			
			dF1_uns = conv_dfilter(F1, imgs_pads, d_uns, PAD=2, gpu=GPU_UNS)
			dF1_s = conv_dfilter(F1, imgs_pads, d_s, PAD=2, gpu=GPU_SUP)

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
		print epoch, batch, err[-1], class_err[-1], np.sum(np.abs(F1)), time.time() - t_start
		savemat('/home/darren/F1.mat', {'F1':F1, 'epoch':epoch, 'class_err':class_err, 'err':err})
		t_start = time.time()
	epoch += 1
sf()