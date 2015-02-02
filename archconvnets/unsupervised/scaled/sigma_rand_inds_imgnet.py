import time
import numpy as np
from archconvnets.unsupervised.cudnn_module import cudnn_module as cm
import archconvnets.unsupervised.sigma31_layers.sigma31_layers as sigma31_layers
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import max_pool_locs, patch_inds, compute_sigma11_gpu
from scipy.io import savemat, loadmat
from scipy.stats import zscore
import random
import copy

N_INDS_KEEP = 10000

conv_block_cuda = cm.conv
F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 128 # batch size
IMG_SZ_CROP = 138 # input image size (px)
IMG_SZ = 138#70#75# # input image size (px)
img_train_offset = 0
PAD = 2

N = 48
n1 = N # L1 filters
n2 = N
n3 = N

s1 = 5
s2 = 5
s3 = 3

N_C = 999 # number of categories

compute_training_sigmas = False
if compute_training_sigmas:
	file_name = '/home/darren/sigmas_imgnet_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat'
	batches = range(1,9100)
else: # compute testing data (patches)
	n_test_batches = 273
	file_name = '/home/darren/patches_imgnet_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat'
	batches = range(9101,9101+n_test_batches)
	
	patches_keep = np.zeros((N_IMGS*n_test_batches, N_INDS_KEEP),dtype='single')
	labels_keep = np.zeros((N_IMGS*n_test_batches),dtype='int')
	
output_sz1 = len(range(0, (IMG_SZ + 2*PAD) - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

np.random.seed(6666)
inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500

sigma31 = np.zeros((N_C, N_INDS_KEEP), dtype='single')
sigma11 = np.zeros((N_INDS_KEEP, N_INDS_KEEP), dtype='single')

imgs_mean = np.load('/export/batch_storage2/batch128_img138_full/batches.meta')['data_mean'][:,np.newaxis]
batch_ind = 0
for batch in batches:
	t_start = time.time()
	z = np.load('/export/batch_storage2/batch128_img138_full/data_batch_' + str(batch))

	################### compute train err
	# load imgs
	x = z['data'] - imgs_mean
	x = x.reshape((3, 138, 138, 128))
	x = x[:,:,:,:N_IMGS]

	labels = np.asarray(z['labels'])[:N_IMGS].astype(int)

	imgs_pad = np.zeros((3, IMG_SZ+2*PAD, IMG_SZ+2*PAD, N_IMGS),dtype='single')
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

	t_forward_start = time.time() - t_forward_start

	t_patch = time.time()
	patches  = patch_inds(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C, inds_keep, warn=False)
	t_patch = time.time() - t_patch
	
	# compute sigma11 & sigma31 or not...
	if compute_training_sigmas:
		t_sigma11 = time.time()
		sigma11 += compute_sigma11_gpu(patches)
		t_sigma11 = time.time() - t_sigma11

		for cat in range(N_C):
			sigma31[cat] += patches[labels == cat].sum(0)
		
		if (batch % 20) == 0:
			print 'saving...'
			savemat(file_name,{'batch': batch,'sigma11':sigma11, 'sigma31':sigma31})
		
		print batch, time.time() - t_start, t_forward_start, t_patch, t_sigma11, file_name
	else:
		patches_keep[batch_ind*N_IMGS:(batch_ind+1)*N_IMGS] = copy.deepcopy(patches)
		labels_keep[batch_ind*N_IMGS:(batch_ind+1)*N_IMGS] = copy.deepcopy(labels)
		
		if (batch % 20) == 0:
			print 'saving...'
			savemat(file_name,{'batch':batch, 'patches':patches_keep,'labels':labels_keep})
			
		print batch, time.time() - t_start, t_forward_start, file_name
	batch_ind += 1

if compute_training_sigmas:
	savemat(file_name,{'batch': batch,'sigma11':sigma11, 'sigma31':sigma31})
else:
	savemat(file_name,{'batch':batch, 'patches':patches_keep,'labels':labels_keep})
