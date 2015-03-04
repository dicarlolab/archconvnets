import time
import numpy as np
from archconvnets.unsupervised.cudnn_module import cudnn_module as cm
import archconvnets.unsupervised.sigma31_layers.sigma31_layers as sigma31_layers
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import max_pool_locs, patch_inds, compute_sigma11_gpu
from scipy.io import savemat, loadmat
from scipy.stats import zscore
import random
import copy

N_INDS_KEEP = 2

conv_block_cuda = cm.conv
F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 10000 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34#70#75# # input image size (px)
img_train_offset = 0
PAD = 2

N = 16
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

np.random.seed(660662)
#inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)

###########
# dense sample points with even spread over F1 values
remainder_inds = np.random.randint(4*5*5*4*3*3*2*2, size=N_INDS_KEEP)
remainder_inds_unraveled = np.asarray(np.unravel_index(remainder_inds, (n2,5,5,n3,3,3,2,2)))

global_inds = np.zeros((12, n1*3*5*5*N_INDS_KEEP),dtype='int')
for f_ind in range(n1*3*5*5):
	f_inds_unraveled = np.asarray(np.unravel_index(f_ind, (n1, 3, 5, 5)))
	global_inds[:, f_ind*N_INDS_KEEP:(f_ind+1)*N_INDS_KEEP] = np.concatenate((np.tile(f_inds_unraveled[:,np.newaxis], (1, N_INDS_KEEP)), remainder_inds_unraveled), axis=0)

inds_keepL1 = np.ravel_multi_index(global_inds, (n1,3,5,5,n2,5,5,n3,3,3,2,2))

###########
# dense sample points with even spread over F2 values
remainder_inds = np.random.randint(3*5*5*4*3*3*2*2, size=N_INDS_KEEP)
remainder_inds_unraveled = np.asarray(np.unravel_index(remainder_inds, (3,5,5,4,3,3,2,2)))

global_inds = np.zeros((12, n2*n1*5*5*N_INDS_KEEP),dtype='int')
for f_ind in range(n2*n1*5*5):
	f_inds_unraveled = np.asarray(np.unravel_index(f_ind, (n2, n1, 5, 5)))
	global_inds[:, f_ind*N_INDS_KEEP:(f_ind+1)*N_INDS_KEEP] = np.concatenate((np.tile(f_inds_unraveled[:,np.newaxis], (1, N_INDS_KEEP)), remainder_inds_unraveled), axis=0)

global_inds = global_inds[[1, 4, 5, 6, 0, 2, 3, 7, 8, 9, 10, 11]] # put F1 indices first
inds_keepL2 = np.ravel_multi_index(global_inds, (n1,3,5,5,n2,5,5,n3,3,3,2,2))

###########
# dense sample points with even spread over F3 values
remainder_inds = np.random.randint(4*3*5*5*5*5*2*2, size=N_INDS_KEEP)
remainder_inds_unraveled = np.asarray(np.unravel_index(remainder_inds, (4,3,5,5,5,5,2,2)))

global_inds = np.zeros((12, n3*n2*3*3*N_INDS_KEEP),dtype='int')
for f_ind in range(n3*n2*3*3):
	f_inds_unraveled = np.asarray(np.unravel_index(f_ind, (n3, n2, 3, 3)))
	global_inds[:, f_ind*N_INDS_KEEP:(f_ind+1)*N_INDS_KEEP] = np.concatenate((np.tile(f_inds_unraveled[:,np.newaxis], (1, N_INDS_KEEP)), remainder_inds_unraveled), axis=0)

global_inds = global_inds[[4, 5, 6, 7, 1, 8, 9, 0, 2, 3, 10, 11]] # put F1 indices first
inds_keepL3 = np.ravel_multi_index(global_inds, (n1,3,5,5,n2,5,5,n3,3,3,2,2))

inds_keep = np.concatenate((inds_keepL1, inds_keepL2, inds_keepL3))

#####
np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500

sigma31 = np.zeros((N_C, len(inds_keep)), dtype='single')
sigma11 = np.zeros((len(inds_keep), len(inds_keep)), dtype='single')

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
for batch in [1,6]:#range(1,7):
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
	conv_output1 = conv_block_cuda(F1, imgs_pad)
	max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(np.single(conv_output1),warn=False)

	max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
	max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

	conv_output2 = conv_block_cuda(F2, max_output1)
	max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(np.single(conv_output2), PAD=2,warn=False)

	max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
	max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

	conv_output3 = conv_block_cuda(F3, max_output2)
	max_output3t, output_switches3_x, output_switches3_y = max_pool_locs(np.single(conv_output3), PAD=2,warn=False)

	output_switches2_x -= PAD
	output_switches2_y -= PAD

	output_switches3_x -= PAD
	output_switches3_y -= PAD

	print time.time() - t_forward_start

	t_patch = time.time()
	patches  = patch_inds(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C, inds_keep, warn=False)
	print time.time() - t_patch
	
	if batch == 6:
		break
		
	t_start = time.time()
	sigma11 += compute_sigma11_gpu(patches)
	print time.time() - t_start

	for cat in range(N_C):
		sigma31[cat] += patches[labels == cat].sum(0)
	print batch, time.time() - t_start


savemat('/home/darren/sigmas_train_test_' + str(N) + '_' + str(N_INDS_KEEP) + '_rand.mat',{'sigma11':sigma11, 'sigma31':sigma31, 'patches':patches,'labels':labels, 'inds_keep': inds_keep,'N_INDS_KEEP':N_INDS_KEEP})
