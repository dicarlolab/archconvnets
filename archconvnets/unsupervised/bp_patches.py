import time
import numpy as np
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import *
from archconvnets.unsupervised.cudnn_module.cudnn_module import *
from scipy.io import savemat, loadmat
from scipy.stats import zscore
import random
import copy

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 128 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34#70#75# # input image size (px)
img_train_offset = 0
PAD = 2

N = 2
n1 = N # L1 filters
n2 = N
n3 = N

s1 = 5
s2 = 5
s3 = 3

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

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500
FL = zscore(FL,axis=None)/500

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
batch = 1
t_start = time.time()
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))

################### compute train err
# load imgs
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_IMGS]

labels = np.asarray(z['labels'])[:N_IMGS].astype(int)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

# forward pass
t_forward_start = time.time()
conv_output1 = conv(F1, imgs_pad)
max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(np.single(conv_output1),warn=False)

max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv(F2, max_output1)
max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(np.single(conv_output2), PAD=2,warn=False)

max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv(F3, max_output2)
max_output3, output_switches3_x, output_switches3_y = max_pool_locs(np.single(conv_output3), PAD=2,warn=False)

output_switches2_x -= PAD
output_switches2_y -= PAD

output_switches3_x -= PAD
output_switches3_y -= PAD

print time.time() - t_forward_start

t_patch = time.time()

EPS = 1e-2
for step in range(100):
	t_patch = time.time()
	
	conv_output1 = conv(F1, imgs_pad)
	max_output1t = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output1[:,np.newaxis])), output_switches1_x, output_switches1_y)
	max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
	max_output1[:,:, PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = np.squeeze(max_output1t)

	conv_output2 = conv(F2, max_output1)
	max_output2t = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output2[:,np.newaxis])), output_switches2_x, output_switches2_y)
	max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
	max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = np.squeeze(max_output2t)

	conv_output3 = conv(F3, max_output2)
	max_output3 = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output3[:,np.newaxis])), output_switches3_x, output_switches3_y)
	pred = np.einsum(np.squeeze(max_output3), [0,1,2,3], FL, [4,1,2,3], [0,4])
	
	#grad_F1 = bp_patch_sigma31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x[:N_C], output_switches3_y[:N_C], output_switches2_x[:N_C], output_switches2_y[:N_C], output_switches1_x[:N_C], output_switches1_y[:N_C], imgs_pad, imgs_pad[:N_C], 1, pred, F1, F2, F3, FL)
	grad_F1 = bp_patch_sigma31_gpu(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x[:N_C], output_switches3_y[:N_C], output_switches2_x[:N_C], output_switches2_y[:N_C], output_switches1_x[:N_C], output_switches1_y[:N_C], imgs_pad, imgs_pad[:N_C], 1, pred, F1, F2, F3, FL)
	F1 -= grad_F1*EPS
	
	'''grad_F2 = bp_patch_sigma31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x[:N_C], output_switches3_y[:N_C], output_switches2_x[:N_C], output_switches2_y[:N_C], output_switches1_x[:N_C], output_switches1_y[:N_C], imgs_pad, imgs_pad[:N_C], 2, pred, F1, F2, F3, FL)
	F2 -= grad_F2*EPS
	
	grad_F3 = bp_patch_sigma31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x[:N_C], output_switches3_y[:N_C], output_switches2_x[:N_C], output_switches2_y[:N_C], output_switches1_x[:N_C], output_switches1_y[:N_C], imgs_pad, imgs_pad[:N_C], 3, pred, F1, F2, F3, FL)
	F3 -= grad_F3*EPS
	
	grad_FL = bp_patch_sigma31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x[:N_C], output_switches3_y[:N_C], output_switches2_x[:N_C], output_switches2_y[:N_C], output_switches1_x[:N_C], output_switches1_y[:N_C], imgs_pad, imgs_pad[:N_C], 4, pred, F1, F2, F3, FL)
	FL -= grad_FL*EPS
	'''
	savemat('/home/darren/F1.mat', {'F1': F1})

	print time.time() - t_patch

