import time
import numpy as np
from archconvnets.unsupervised.cudnn_module import cudnn_module as cm
import archconvnets.unsupervised.sigma31_layers.sigma31_layers as sigma31_layers
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import max_pool_locs, patch_inds, compute_sigma11_lin_gpu
from scipy.io import savemat, loadmat
from scipy.stats import zscore
import random
import copy
import os

N_INDS_KEEP = 1000

conv_block_cuda = cm.conv
F1_scale = 0.0001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01


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

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))

N_PAIRS = 0.5*(N_INDS_KEEP-1)*N_INDS_KEEP + N_INDS_KEEP

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
batch = 6

z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))

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

for step in range(500):
	np.random.seed(6666 + step)
	#inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)
	N_REP = 10
	inds_new = np.zeros((12, n1*3*s1*s1*N_REP), dtype='int')
	ind = 0

	np.random.seed(6666 + step)
	for rep in range(N_REP):
		for f1 in range(n1):
			for f0 in range(3):
				for a1_x in range(s1):
					for a1_y in range(s2):
						inds_new[0,ind] = f1
						inds_new[1,ind] = f0
						inds_new[2,ind] = a1_x
						inds_new[3,ind] = a1_y
						inds_new[4,ind] = np.random.randint(n2)
						inds_new[5,ind] = np.random.randint(s2)
						inds_new[6,ind] = np.random.randint(s2)
						inds_new[7,ind] = np.random.randint(n3)
						inds_new[8,ind] = np.random.randint(s3)
						inds_new[9,ind] = np.random.randint(s3)
						inds_new[10,ind] = np.random.randint(max_output_sz3)
						inds_new[11,ind] = np.random.randint(max_output_sz3)
						ind += 1

	inds_keep = np.ravel_multi_index(inds_new, (n1,3,s1,s1,n2,s2,s2,n3,s3,s3,max_output_sz3,max_output_sz3))
	break
	print len(inds_keep), len(np.unique(inds_keep))

	t_patch = time.time()
	patches  = patch_inds(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C, inds_keep, warn=False)
	print time.time() - t_patch

	t_start = time.time()
	sigma11 = compute_sigma11_lin_gpu(patches)
	print time.time() - t_start

	sigma31 = np.zeros((N_C, len(inds_keep)), dtype='single')
	for cat in range(N_C):
		sigma31[cat] += patches[labels == cat].sum(0)
	print step, time.time() - t_start

	if step == 0:
		savemat('/export/imgnet_storage_full/sigma31_inds/sigmas_' + str(N) + '_' + str(N_INDS_KEEP) + '_' + str(step) + '.mat',{'sigma11':sigma11, 'sigma31':sigma31, 'patches':patches,'labels':labels,'N_IMGS':N_IMGS})
	else:
		savemat('/export/imgnet_storage_full/sigma31_inds/sigmas_' + str(N) + '_' + str(N_INDS_KEEP) + '_' + str(step) + '.mat',{'sigma11':sigma11, 'sigma31':sigma31,'N_IMGS':N_IMGS})
	
