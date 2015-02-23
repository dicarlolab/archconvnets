from archconvnets.unsupervised.conv import conv_block
import time
import numpy as np
from archconvnets.unsupervised.pool_inds_py import max_pool_locs
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from archconvnets.unsupervised.cudnn_module.cudnn_module import *
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import *

conv_block_cuda = conv

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 5 # batch size
N_TEST_IMGS = N_IMGS #N_SIGMA_IMGS #128*2
N_SIGMA_IMGS = N_IMGS
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
F1_init = copy.deepcopy(F1)
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F2_init = copy.deepcopy(F2)
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
F3_init = copy.deepcopy(F3)
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))
FL_init = copy.deepcopy(FL)
	
F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500
FL = zscore(FL,axis=None)/500
imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']


y = loadmat('/home/darren/sigma31_dbg.mat')
sigma31 = y['sigma31']
#sigma31 = sigma31.transpose((0,2,1,3,4,5,6,7,8,9,10,11,12))

# 10, n1, 3, s1, s1, n2, s2, s2, n3, s3, s3, sz, sz

sigma31_L1 = sigma31
sigma31_L2 = sigma31
sigma31_L3 = sigma31
sigma31_LF = sigma31

##################
# load test imgs into buffers
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_TEST_IMGS]

l = np.zeros((N_TEST_IMGS, N_C),dtype='int')
l[np.arange(N_TEST_IMGS),np.asarray(z['labels'])[:N_TEST_IMGS].astype(int)] = 1
Y_test = np.double(l.T)
img_cats = np.asarray(z['labels'])[:N_IMGS].astype(int)

i_ind = 1
j_ind = 1
k_ind = 1
l_ind = 0

step = 0

def f(y):
	F1[i_ind, j_ind, k_ind, l_ind] = y
		
	FLr = FL.reshape((N_C, n3*max_output_sz3**2))
	
	########################## compute test err
	
	imgs_pad_batch = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
	imgs_pad_batch[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP,step*N_IMGS:(step+1)*N_IMGS]
	imgs_pad_batch = np.ascontiguousarray(imgs_pad_batch.transpose((3,0,1,2)))
	
	# forward pass init filters
	conv_output1 = conv(F1, imgs_pad_batch)
	max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)
	max_output1t, pool1_patches = max_pool_locs_alt_patches(conv_output1, output_switches1_x, output_switches1_y, imgs_pad_batch, s1)
	max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
	max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

	conv_output2 = conv(F2, max_output1)
	max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)
	max_output2t, pool2_patches = max_pool_locs_alt_patches(conv_output2, output_switches2_x, output_switches2_y, max_output1, s2)
	max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
	max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

	conv_output3 = conv(F3, max_output2)
	max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)
	max_output3, pool3_patches = max_pool_locs_alt_patches(conv_output3, output_switches3_x, output_switches3_y, max_output2, s3)

	pred = np.dot(FLr, max_output3.reshape((N_TEST_IMGS, n3*max_output_sz3**2)).T)
	err = np.sum((pred - Y_test)**2)
	return np.sum(err)
	
#sigma31_LF = sigma31.mean(2)#.mean(2).mean(2).mean(2).mean(2).mean(2).mean(3).mean(3)
#sigma31_LF = sigma31_LF.reshape((N_C, n1, 1, s1, s1, n2, s2, s2, n3, s3, s3, sigma31.shape[-1], sigma31.shape[-1]))
set_sigma_buffer(np.ascontiguousarray(np.single(sigma31_L1)), 1, 0)
set_sigma_buffer(np.ascontiguousarray(np.single(sigma31_L2)), 2, 1)
set_sigma_buffer(np.ascontiguousarray(np.single(sigma31_L3)), 3, 2)
set_sigma_buffer(np.ascontiguousarray(np.single(sigma31_LF)), 4, 3)

def g(y):
	F1[i_ind, j_ind, k_ind, l_ind] = y
	FLr = FL.reshape((N_C, n3*max_output_sz3**2))
	########################## compute test err
	
	imgs_pad_batch = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
	imgs_pad_batch[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP,step*N_IMGS:(step+1)*N_IMGS]
	imgs_pad_batch = np.ascontiguousarray(imgs_pad_batch.transpose((3,0,1,2)))
	
	# forward pass init filters
	conv_output1 = conv(F1, imgs_pad_batch)
	max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)
	max_output1t, pool1_patches = max_pool_locs_alt_patches(conv_output1, output_switches1_x, output_switches1_y, imgs_pad_batch, s1)
	max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
	max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

	conv_output2 = conv(F2, max_output1)
	max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)
	max_output2t, pool2_patches = max_pool_locs_alt_patches(conv_output2, output_switches2_x, output_switches2_y, max_output1, s2)
	max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
	max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

	conv_output3 = conv(F3, max_output2)
	max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)
	max_output3, pool3_patches = max_pool_locs_alt_patches(conv_output3, output_switches3_x, output_switches3_y, max_output2, s3)

	pred = np.dot(FLr, max_output3.reshape((N_TEST_IMGS, n3*max_output_sz3**2)).T)
	pred[img_cats, range(N_IMGS)] -= 1 ############ backprop supervised term
	pred_ravel = pred.ravel()
	
	########## FL deriv wrt cat_, f3_, z1_, z2_
	grad_FL_uns = np.einsum(pred,[4,0],max_output3,range(4),[4,1,2,3]) 
	
	########## F3 deriv wrt f3_, a3_x_, a3_y_, f2_
	pred_deriv = np.einsum(pool3_patches, range(7), FL, [7, 4, 5, 6], [7, 0, 4, 1, 2, 3])
	grad_L3_uns = np.einsum(pred, [0,1], pred_deriv, range(6), [2,3,4,5])
	
	########## F2 deriv wrt f2_, a2_x_, a2_y_, f1_
	# ravel together all the patches to reduce the needed convolution function calls
	pool2_derivt = pool2_patches.reshape((N_IMGS*n1*s2*s2, n2, max_output_sz2-2*PAD, max_output_sz2-2*PAD))
	pool2_deriv = np.zeros((N_IMGS*n1*s2*s2, n2, max_output_sz2, max_output_sz2),dtype='single')
	pool2_deriv[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = pool2_derivt

	pool2_deriv = np.ascontiguousarray(pool2_deriv.transpose((1,0,2,3))[:,:,np.newaxis])
	F3c = np.ascontiguousarray(F3.transpose((1,0,2,3))[:,:,np.newaxis])

	max_output3t_accum = np.zeros((N_IMGS, n2, n1, s2, s2, n3*max_output_sz3**2),dtype='single')
	for f2_ in range(n2):
		conv_output3_deriv = conv(F3c[f2_], pool2_deriv[f2_])
		conv_output3_deriv = conv_output3_deriv.reshape((N_IMGS, n1*s2*s2, n3, output_sz3, output_sz3))
		
		max_output3t = max_pool_locs_alt(conv_output3_deriv, output_switches3_x, output_switches3_y)
		max_output3t_accum[:,f2_] = max_output3t.reshape((N_IMGS, n1, s2, s2, n3*max_output_sz3**2))
		
	pred_deriv = np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4)))

	pred_deriv = pred_deriv.reshape((N_C*N_IMGS, n2, n1, s2, s2)).transpose((1,2,3,0,4))
	grad_L2_uns = np.dot(pred_ravel, pred_deriv) 

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
		
		max_output2t = max_pool_locs_alt(conv_output2_deriv, output_switches2_x, output_switches2_y)
		max_output2 = np.zeros((N_IMGS, 3*s1*s1, n2, max_output_sz2, max_output_sz2),dtype='single')
		max_output2[:,:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t
		max_output2 = max_output2.reshape((N_IMGS*3*s1*s1, n2, max_output_sz2, max_output_sz2))
		
		conv_output3_deriv = conv(F3, max_output2)
		conv_output3_deriv = conv_output3_deriv.reshape((N_IMGS, 3*s1*s1, n3, output_sz3, output_sz3))
		
		max_output3t = max_pool_locs_alt(conv_output3_deriv, output_switches3_x, output_switches3_y)
		max_output3t_accum[:,f1_] = max_output3t.reshape((N_IMGS, 3, s1, s1, n3*max_output_sz3**2))
		
	pred_deriv = np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4))) # sum across f3,z1,z2

	pred_deriv = pred_deriv.reshape((N_C*N_IMGS, n1, 3, s1, s1)).transpose((1,2,3,0,4))
	grad_L1_uns = np.dot(pred_ravel, pred_deriv)  # sum across imgs and predictions (J_c)
	
	return grad_L1_uns[i_ind,j_ind,k_ind,l_ind]


eps = np.sqrt(np.finfo(np.float).eps)*2e1
y = 1e-4*FL_init[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps); print gt, gtx, gtx/gt
y = 1e-5*FL_init[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps); print gt, gtx, gtx/gt
y = -1e-4*FL_init[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps); print gt, gtx, gtx/gt


