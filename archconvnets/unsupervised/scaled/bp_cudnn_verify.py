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
N_IMGS = 256 # batch size
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
img_cats = np.asarray(z['labels'])[:N_IMGS].astype(int)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP]
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

conv_output1 = conv(F1, imgs_pad)
max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)
max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv(F2, max_output1)
max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2, PAD=2)
max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv(F3, max_output2)
max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3, PAD=2)

#output_switches2_x -= PAD
#output_switches2_y -= PAD

#output_switches3_x -= PAD
#output_switches3_y -= PAD

sigma31 = s31_full_gpu(output_switches3_x - PAD, output_switches3_y - PAD, output_switches2_x - PAD, output_switches2_y - PAD, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C)

#########
sigma31_F1 = sigma31*F1.reshape((1, n1, 3, s1, s1, 1, 1, 1, 1, 1,1,1,1))
sigma31_F2 = sigma31_F1*F2.transpose((1,0,2,3)).reshape((1, n1, 1, 1, 1, n2, s2, s2, 1, 1,1,1,1))
sigma31_F3 = sigma31_F2*F3.transpose((1,0,2,3)).reshape((1, 1, 1, 1, 1, n2, 1, 1, n3, s3, s3, 1, 1))
sigma31_F3 = sigma31_F3[6].reshape((n1*3*(s1**2)*n2*(s2**2), n3, s3**2, 2, 2)).sum(0).sum(1)[np.newaxis]
print np.isclose(sigma31_F3, max_output3[0][np.newaxis]).sum() / np.single(np.prod(sigma31_F3.shape))
#####

set_sigma_buffer(sigma31, 1, 0)

set_filter_buffers(F1,F2,F3,FL,0)
einsum_deriv_gpu(1,1,1,0) # deriv, l1

FLr = FL.reshape((N_C, n3*max_output_sz3**2))

max_output1t, pool1_patches = max_pool_locs_alt_patches(conv_output1, output_switches1_x, output_switches1_y, imgs_pad, s1)

pred = np.dot(FLr, max_output3.reshape((N_IMGS, n3*max_output_sz3**2)).T)

#pred = np.zeros_like(pred)
#pred[img_cats, range(N_IMGS)] = 1 ############ backprop supervised term

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
grad_L1_uns = np.dot(pred_ravel, pred_deriv) # sum across imgs and predictions (J_c)

derivc = einsum_return(1,0) # [prediction each mean category makes for each category, category f1 inds]
'''
#### check for einsum_return -- sigma31 * FL32
# verifies einsum_deriv_gpu
sigma31_F2 = sigma31*F2.transpose((1,0,2,3)).reshape((1, n1, 1, 1, 1, n2, s2, s2, 1, 1,1,1,1))
sigma31_F3 = sigma31_F2*F3.transpose((1,0,2,3)).reshape((1, 1, 1, 1, 1, n2, 1, 1, n3, s3, s3, 1, 1))
derivc_n = np.einsum(sigma31_F3, range(13), FL, [14, 8, 11, 12], [0, 14, 1,2,3,4])
print np.isclose(derivc_n, derivc).sum() / np.single(np.prod(derivc.shape))

#### check that max output 3 accum matches derivc
# verifies max_pool_locs_alt, max_pool_locs_alt_patches
max_output3t_accum_n = sigma31_F3.sum(5).sum(5).sum(5).sum(6).sum(6).reshape((N_C, n1, 3, s1, s1, n3*max_output_sz3**2))
print np.isclose(max_output3t_accum[0], max_output3t_accum_n[6]).sum()/np.single(np.prod(max_output3t_accum.shape))

##### check that pred_deriv == derivc
# verifies functions above, again
pred_deriv = np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4))) # sum across f3,z1,z2
print np.isclose(np.squeeze(pred_deriv), derivc[:,6]).sum()/np.single(np.prod(pred_deriv.shape))

print np.isclose(np.squeeze(pred_deriv.sum(0)), derivc.sum(0).sum(0)).sum()/np.single(np.prod(pred_deriv.sum(0).shape))
pred_deriv = pred_deriv.reshape((N_C*N_IMGS, n1, 3, s1, s1)).transpose((1,2,3,0,4))
print np.isclose(pred_deriv[:,:,:,6], grad_L1_uns).sum()/np.single(np.prod(grad_L1_uns.shape))
print np.isclose(grad_L1_uns, derivc[6].sum(0)).sum()/np.single(np.prod(grad_L1_uns.shape))'''
print np.isclose(grad_L1_uns, derivc.sum(0)).sum()/np.single(np.prod(grad_L1_uns.shape))
grad = grad_L1_uns - derivc.sum(0)

################

pred = np.dot(FLr, max_output3.reshape((N_IMGS, n3*max_output_sz3**2)).T)

pred[img_cats, range(N_IMGS)] -= 1 ############ backprop supervised term

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
grad_L1_uns = np.dot(pred_ravel, pred_deriv) # sum across imgs and predictions (J_c)

print np.isclose(grad_L1_uns, grad).sum()/np.single(np.prod(grad.shape))
