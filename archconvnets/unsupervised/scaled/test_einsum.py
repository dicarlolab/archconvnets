from archconvnets.unsupervised.conv import conv_block
import time
import numpy as np
from archconvnets.unsupervised.pool_inds_py import max_pool_locs
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
#from archconvnets.unsupervised.cudnn_module.cudnn_module import *
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import *

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3


POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
IMG_SZ_CROP = 28 # input image size (px)
IMG_SZ = 32 # input image size (px)
img_train_offset = 2
PAD = 2

N = 8
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

y = loadmat('/home/darren/sigma31_dbg8.mat')
#y = loadmat('/home/darren/sigma31_dbg.mat')
sigma31 = y['sigma31']
#sigma31 = sigma31.transpose((0,2,1,3,4,5,6,7,8,9,10,11,12))

# 10, n1, 3, s1, s1, n2, s2, s2, n3, s3, s3, sz, sz


sigma31_L1 = sigma31.mean(-1).mean(-1).mean(-1).mean(-1).mean(-1).mean(-1).mean(-1)
sigma31_L1 = sigma31_L1.reshape((N_C, n1, 3, s1, s1, n2, 1, 1, 1, 1, 1, 1, 1))

sigma31_L2 = sigma31.mean(3).mean(3).mean(-1).mean(-1).mean(-1).mean(-1).mean(-1)
sigma31_L2 = sigma31_L2.reshape((N_C, n1, 3, 1, 1, n2, s2, s2, 1, 1, 1, 1, 1))

sigma31_LF = sigma31.mean(1).mean(2).mean(2).mean(2).mean(2).mean(2).mean(3).mean(3)
sigma31_LF = sigma31_LF.reshape((N_C, 1, 3, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))

sigma31_L3 = sigma31.mean(-1).mean(-1).mean(5).mean(5).mean(5)
sigma31_L3 = sigma31_L3.reshape((N_C, n1, 3, s1, s1, 1, 1, 1, n3, s3, s3, 1, 1))



set_sigma_buffer(np.ascontiguousarray(np.single(sigma31_L1)), 1, 0)
set_sigma_buffer(np.ascontiguousarray(np.single(sigma31_L2)), 2, 1)
set_sigma_buffer(np.ascontiguousarray(np.single(sigma31_L3)), 3, 2)
set_sigma_buffer(np.ascontiguousarray(np.single(sigma31_LF)), 4, 3)

set_filter_buffers(F1,F2,F3,FL,0)
set_filter_buffers(F1,F2,F3,FL,1)
set_filter_buffers(F1,F2,F3,FL,2)
set_filter_buffers(F1,F2,F3,FL,3)

einsum_deriv_gpu(0,1,0,0) # pred, l1
einsum_deriv_gpu(1,1,1,0) # deriv, l1

einsum_deriv_gpu(0,2,0,1) # pred, l2
einsum_deriv_gpu(2,2,1,1) # deriv, l2

einsum_deriv_gpu(0,3,0,2) # pred, l3
einsum_deriv_gpu(3,3,1,2) # deriv, l3

einsum_deriv_gpu(0,4,0,3) # pred, fl
einsum_deriv_gpu(4,4,1,3) # deriv, fl

###############
FLt = FL.reshape((N_C, 1, 1, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))

F21 = F1[:,:,:,:,np.newaxis,np.newaxis,np.newaxis] * F2.transpose((1,0,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
F21 = F21.reshape((1, n1, 3, s1, s1, n2, s2, s2, 1, 1, 1, 1, 1))
F321 = F21 * F3.transpose((1,0,2,3)).reshape((1, 1, 1, 1, 1, n2, 1, 1, n3, s3, s3, 1, 1))
FL321 = F321 * FL.reshape((N_C, 1, 1, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))

sigma_inds = [0,2,3,4,5,6,7,8,9,10,11,12,13]
F_inds = [1,2,3,4,5,6,7,8,9,10,11,12,13]


############################################## F1 deriv
F32 = F2[np.newaxis,:,:,:,:,np.newaxis,np.newaxis] * F3[:,:,np.newaxis,np.newaxis,np.newaxis]
F32 = F32.transpose((2,1,3,4,0,5,6))
F32 = F32[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:,:,:,:,:,:,np.newaxis,np.newaxis]
FL32 = FLt * F32


derivc = np.einsum(sigma31_L1, sigma_inds, FL32, F_inds, [1,0,2,3,4,5])
predc = np.einsum(sigma31_L1, sigma_inds, FL321, F_inds, [1,0])

predcn = einsum_return(0,0)
derivcn = einsum_return(1,0)

print np.isclose(predc.ravel(), predcn.ravel()).sum()/np.single(np.prod(predc.shape))
print np.isclose(derivc.ravel(), derivcn.ravel()).sum()/np.single(np.prod(derivc.shape))


############################################# F2 deriv
F31 = np.tensordot(F1, F3, 0).transpose((0,1,2,3,5,4,6,7))
F31 = F31.reshape((1,n1, 3, s1, s1, n2, 1,1, n3, s3, s3, 1, 1))
FL31 = FLt * F31

derivc = np.einsum(sigma31_L2, sigma_inds, FL31, F_inds, [1,0,6,2,7,8])
predc = np.einsum(sigma31_L2, sigma_inds, FL321, F_inds, [1,0])

predcn = einsum_return(0,1)
derivcn = einsum_return(1,1)

print np.isclose(predc.ravel(), predcn.ravel()).sum()/np.single(np.prod(predc.shape))
print np.isclose(derivc.ravel(), derivcn.ravel()).sum()/np.single(np.prod(derivc.shape))


############################################# F3 deriv
FL21 = FLt * F21
	
derivc = np.einsum(sigma31_L3, sigma_inds, FL21, F_inds, [1,0,9,6,10,11])
predc = np.einsum(sigma31_L3, sigma_inds, FL321, F_inds, [1,0])

predcn = einsum_return(0,2)
derivcn = einsum_return(1,2)

print np.isclose(predc.ravel(), predcn.ravel()).sum()/np.single(np.prod(predc.shape))
print np.isclose(derivc.ravel(), derivcn.ravel()).sum()/np.single(np.prod(derivc.shape))  #######################


####################################### FL deriv wrt cat_, f3_, z1_, z2_
derivc = np.einsum(sigma31_LF, sigma_inds, F321, sigma_inds, [0,9,12,13])[np.newaxis]
predc = np.einsum(sigma31_LF, sigma_inds, FL321, F_inds, [1,0])

predcn = einsum_return(0,3)
derivcn = einsum_return(1,3)

print np.isclose(predc.ravel(), predcn.ravel()).sum()/np.single(np.prod(predc.shape))
print np.isclose(derivc.ravel(), derivcn.ravel()).sum()/np.single(np.prod(derivc.shape))  #######################
