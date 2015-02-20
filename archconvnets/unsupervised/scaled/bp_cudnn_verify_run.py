from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import numpy as np
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import *
import copy
from scipy.stats import zscore
import random

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 128 # batch size
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

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

np.random.seed(6666)
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
x = x[:,:,:,:N_IMGS]

labels = np.asarray(z['labels'])[:N_IMGS].astype(int)
l = np.zeros((N_IMGS, N_C),dtype='int')
l[np.arange(N_IMGS),np.asarray(z['labels'])[:N_IMGS].astype(int)] = 1
Y = np.double(l.T)
img_cats = np.asarray(z['labels'])[:N_IMGS].astype(int)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP]
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

conv_output1 = conv(F1, imgs_pad)
max_output1t, output_switches1_x_init, output_switches1_y_init = max_pool_locs(conv_output1)
max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv(F2, max_output1)
max_output2t, output_switches2_x_init, output_switches2_y_init = max_pool_locs(conv_output2, PAD=2)
max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv(F3, max_output2)
max_output3, output_switches3_x_init, output_switches3_y_init = max_pool_locs(conv_output3, PAD=2)


sigma31 = s31_full_gpu(output_switches3_x_init - PAD, output_switches3_y_init - PAD, output_switches2_x_init - PAD, output_switches2_y_init - PAD, output_switches1_x_init, output_switches1_y_init, s1, s2, s3, labels, imgs_pad, N_C)

set_sigma_buffer(sigma31, 1, 0)

err = []; class_err = []

for step in range(10):
	t_start = time.time()
	set_filter_buffers(F1,F2,F3,FL,0)
	einsum_deriv_gpu(1,1,1,0) # deriv, l1

	FLr = FL.reshape((N_C, n3*max_output_sz3**2))

	conv_output1 = conv(F1, imgs_pad)
	max_output1t, pool1_patches = max_pool_locs_alt_patches(conv_output1, output_switches1_x_init, output_switches1_y_init, imgs_pad, s1)
	max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
	max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

	conv_output2 = conv(F2, max_output1)
	max_output2t, pool2_patches = max_pool_locs_alt_patches(conv_output2, output_switches2_x_init, output_switches2_y_init, max_output1, s2)
	max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
	max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

	conv_output3 = conv(F3, max_output2)
	max_output3, pool3_patches = max_pool_locs_alt_patches(conv_output3, output_switches3_x_init, output_switches3_y_init, max_output2, s3)


	pred = np.dot(FLr, max_output3.reshape((N_IMGS, n3*max_output_sz3**2)).T)
	
	err.append(np.sum((pred - Y)**2)/N_IMGS)
	class_err.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y,axis=0)))/N_IMGS)
	
	#pred[img_cats, range(N_IMGS)] -= 1 ############ backprop supervised term

	pred_ravel = pred.ravel()

	########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_
	# ravel together all the patches to reduce the needed convolution function calls
	pool1_derivt = pool1_patches.reshape((N_IMGS*3*s1*s1, n1, max_output_sz1-2*PAD, max_output_sz1-2*PAD))
	pool1_deriv = np.zeros((N_IMGS*3*s1*s1, n1, max_output_sz1, max_output_sz1),dtype='single')
	pool1_deriv[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = pool1_derivt

	pool1_deriv = np.ascontiguousarray(pool1_deriv.transpose((1,0,2,3))[:,:,np.newaxis])
	F2c = np.ascontiguousarray(F2.transpose((1,0,2,3))[:,:,np.newaxis])

	max_output3t_accum = np.zeros((N_IMGS, n1, 3, s1, s1, n3*max_output_sz3**2),dtype='single')
	for f1_ in range(n1):
		conv_output2_deriv = conv(F2c[f1_], pool1_deriv[f1_])
		conv_output2_deriv = conv_output2_deriv.reshape((N_IMGS, 3*s1*s1, n2, output_sz2, output_sz2))
		
		max_output2t = max_pool_locs_alt(conv_output2_deriv, output_switches2_x_init, output_switches2_y_init)
		max_output2 = np.zeros((N_IMGS, 3*s1*s1, n2, max_output_sz2, max_output_sz2),dtype='single')
		max_output2[:,:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t
		max_output2 = max_output2.reshape((N_IMGS*3*s1*s1, n2, max_output_sz2, max_output_sz2))
		
		conv_output3_deriv = conv(F3, max_output2)
		conv_output3_deriv = conv_output3_deriv.reshape((N_IMGS, 3*s1*s1, n3, output_sz3, output_sz3))
		
		max_output3t = max_pool_locs_alt(conv_output3_deriv, output_switches3_x_init, output_switches3_y_init)
		max_output3t_accum[:,f1_] = max_output3t.reshape((N_IMGS, 3, s1, s1, n3*max_output_sz3**2))
		
	pred_deriv = np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4))) # sum across f3,z1,z2

	pred_deriv = pred_deriv.reshape((N_C*N_IMGS, n1, 3, s1, s1)).transpose((1,2,3,0,4))
	grad_L1_uns = np.dot(pred_ravel, pred_deriv) # sum across imgs and predictions (J_c)

	derivc = einsum_return(1,0) # [prediction each mean category makes for each category, category f1 inds]
	#derivc = np.zeros_like(derivc)

	grad = grad_L1_uns - derivc[range(N_C),range(N_C)].sum(0)
	F1 -= 1e-1*grad
	print err[-1], class_err[-1], time.time() - t_start

