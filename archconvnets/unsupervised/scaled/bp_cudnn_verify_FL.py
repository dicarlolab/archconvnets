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


sigma31 = s31_full_gpu(output_switches3_x - PAD, output_switches3_y - PAD, output_switches2_x - PAD, output_switches2_y - PAD, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C)


set_sigma_buffer(sigma31, 1, 0)

set_filter_buffers(F1,F2,F3,FL,0)
einsum_deriv_gpu(4,1,1,0) # deriv, l1

FLr = FL.reshape((N_C, n3*max_output_sz3**2))

max_output1t, pool1_patches = max_pool_locs_alt_patches(conv_output1, output_switches1_x, output_switches1_y, imgs_pad, s1)
max_output2t, pool2_patches = max_pool_locs_alt_patches(conv_output2, output_switches2_x, output_switches2_y, max_output1, s2)
max_output3, pool3_patches = max_pool_locs_alt_patches(conv_output3, output_switches3_x, output_switches3_y, max_output2, s3)

pred = np.dot(FLr, max_output3.reshape((N_IMGS, n3*max_output_sz3**2)).T)

#pred = np.zeros_like(pred)
#pred[img_cats, range(N_IMGS)] = 1 ############ backprop supervised term

pred_ravel = pred.ravel()
########## FL deriv wrt cat_, f3_, z1_, z2_
#grad_FL_uns = np.tile((pred[:,:,np.newaxis,np.newaxis,np.newaxis]*max_output3[np.newaxis]).sum(0).sum(0)[np.newaxis], (N_C,1,1,1)) 
grad_FL_uns = np.einsum(pred,[4,0],max_output3,range(4),[4,1,2,3])

derivc = einsum_return(1,0) # [prediction each mean category makes for each category, category f1 inds]

grad = grad_FL_uns - derivc

################

pred = np.dot(FLr, max_output3.reshape((N_IMGS, n3*max_output_sz3**2)).T)

pred[img_cats, range(N_IMGS)] -= 1 ############ backprop supervised term

pred_ravel = pred.ravel()


########## FL deriv wrt cat_, f3_, z1_, z2_
#grad_FL_uns = np.tile((pred[:,:,np.newaxis,np.newaxis,np.newaxis]*max_output3[np.newaxis]).sum(0).sum(0)[np.newaxis], (N_C,1,1,1)) 
grad_FL_uns = np.einsum(pred,[4,0],max_output3,range(4),[4,1,2,3])

print np.isclose(grad_FL_uns, grad).sum()/np.single(np.prod(grad.shape))
