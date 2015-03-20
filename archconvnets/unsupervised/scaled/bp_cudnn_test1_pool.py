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

#kernprof -l bp_cudnn_test1_pool.py
#python -m line_profiler bp_cudnn_test1_pool.py.lprof  > p
#@profile
#def sf():
F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

EPS = 1e-4
eps_F1 = EPS
eps_F2 = EPS
eps_F3 = EPS
eps_FL = EPS

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 50 # batch size
N_TEST_IMGS = N_IMGS #N_SIGMA_IMGS #128*2
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34 # input image size (px)
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

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

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
		for s in range(200):
			# forward pass
			conv_output1 = conv(F1, imgs_pad[s*N_IMGS:(s+1)*N_IMGS])
			max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)
			max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
			max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t
			conv_output2 = conv(F2, max_output1,warn=False)
			max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)
			max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
			max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t
			conv_output3 = conv(F3, max_output2,warn=False)
			max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)
			
			###### ravel together categories and imgs, replicate across imgs (FL weights) or categories (switches) as necessary to match dims.
			
			pred_uns2 = np.einsum(FL**2, range(4), max_output3, [4,1,2,3], [0,4,1,2,3]).reshape((N_C*N_IMGS, n3, max_output_sz3, max_output_sz3)) # ravel together categories and imgs
			
			Ys_unravel = Y_test[:,s*N_IMGS:(s+1)*N_IMGS]
			Ys = Ys_unravel.reshape((N_C*N_IMGS, 1, 1, 1))
			
			imgs_pads = np.tile(imgs_pad[s*N_IMGS:(s+1)*N_IMGS],(N_C,1,1,1,1)).reshape((N_C*N_IMGS, 3, IMG_SZ, IMG_SZ))
			
			# each category's predictions are weighted differently, but they all use the same switches
			output_switches3_x = np.tile(output_switches3_x,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n3, max_output_sz3, max_output_sz3))
			output_switches3_y = np.tile(output_switches3_y,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n3, max_output_sz3, max_output_sz3))
			
			max_output2 = np.tile(max_output2,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n2, max_output_sz2, max_output_sz2))
			output_switches2_x = np.tile(output_switches2_x,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n2, max_output_sz2 - 2*PAD, max_output_sz2 - 2*PAD))
			output_switches2_y = np.tile(output_switches2_y,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n2, max_output_sz2 - 2*PAD, max_output_sz2 - 2*PAD))
			
			max_output1 = np.tile(max_output1,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n1, max_output_sz1, max_output_sz1))
			output_switches1_x = np.tile(output_switches1_x,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n1, max_output_sz1 - 2*PAD, max_output_sz1 - 2*PAD))
			output_switches1_y = np.tile(output_switches1_y,(N_C,1,1,1,1)).reshape((N_C*N_IMGS, n1, max_output_sz1 - 2*PAD, max_output_sz1 - 2*PAD))
			
			FL_rep_imgs = np.tile(FL,(N_IMGS,1,1,1,1)).transpose((1,0,2,3,4)).reshape((N_C*N_IMGS, n3, max_output_sz3, max_output_sz3))
			
			######## gradients:
			
			## FL
			pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4])
			dFL_uns = np.einsum(pred, [0,1], max_output3, [1,2,3,4], [0,2,3,4])
			dFL_uns_1 = np.einsum(Ys_unravel, [0,1], max_output3, [1,2,3,4], [0,2,3,4])
			
			### F3
			pred_uns2_unpool = unpool(pred_uns2, output_switches3_x, output_switches3_y, conv_output3.shape[2])
			FL_max = unpool(FL_rep_imgs, output_switches3_x, output_switches3_y, conv_output3.shape[2])
			
			dF3_uns = conv_dfilter(F3, max_output2, pred_uns2_unpool)
			dF3_uns_1 = conv_dfilter(F3, max_output2, Ys*FL_max)
			
			#### F2
			dconv_output2 = conv_ddata(F3, max_output2, pred_uns2_unpool)
			dconv_output2_1 = conv_ddata(F3, max_output2, Ys*FL_max)
			
			dconv_output2 = dconv_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD]
			dconv_output2_1 = dconv_output2_1[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD]
			
			dconv_output2 = unpool(dconv_output2, output_switches2_x, output_switches2_y, conv_output2.shape[2],warn=False)
			dconv_output2_1 = unpool(dconv_output2_1, output_switches2_x, output_switches2_y, conv_output2.shape[2],warn=False)
			
			dF2_uns = conv_dfilter(F2, max_output1, dconv_output2)
			dF2_uns_1 = conv_dfilter(F2, max_output1, dconv_output2_1)
			
			## F1
			dconv_output1 = conv_ddata(F2, max_output1, dconv_output2)
			dconv_output1_1 = conv_ddata(F2, max_output1, dconv_output2_1)
			
			dconv_output1 = dconv_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD]
			dconv_output1_1 = dconv_output1_1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD]
			
			dconv_output1 = unpool(dconv_output1, output_switches1_x, output_switches1_y, conv_output1.shape[2],warn=False)
			dconv_output1_1 = unpool(dconv_output1_1, output_switches1_x, output_switches1_y, conv_output1.shape[2],warn=False)
			
			dF1_uns = conv_dfilter(F1, imgs_pads, dconv_output1)
			dF1_uns_1 = conv_dfilter(F1, imgs_pads, dconv_output1_1)
			
			grad_F1 = dF1_uns - dF1_uns_1
			grad_F2 = dF2_uns - dF2_uns_1
			grad_F3 = dF3_uns - dF3_uns_1
			grad_FL = dFL_uns - dFL_uns_1
				
			F1 -= grad_F1*EPS / N_IMGS
			F2 -= grad_F2*EPS / N_IMGS
			F3 -= grad_F3*EPS / N_IMGS
			FL -= grad_FL*EPS / N_IMGS
		
		err.append(np.mean((pred - Ys)**2))
		class_err.append(1-(pred.argmax(0) == labels[s*N_IMGS:(s+1)*N_IMGS]).mean())
		print epoch, batch, err[-1], class_err[-1], np.sum(np.abs(F1)), time.time() - t_start
		savemat('/home/darren/F1.mat', {'F1':F1, 'epoch':epoch, 'class_err':class_err, 'err':err})
		t_start = time.time()
	epoch += 1
sf()