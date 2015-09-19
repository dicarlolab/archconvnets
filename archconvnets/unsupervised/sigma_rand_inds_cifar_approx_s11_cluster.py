import time
import numpy as np
from archconvnets.unsupervised.cudnn_module import cudnn_module as cm
import archconvnets.unsupervised.sigma31_layers.sigma31_layers as sigma31_layers
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import max_pool_locs, patch_inds_addresses, compute_sigma11_gpu
from scipy.io import savemat, loadmat
from scipy.stats import zscore, pearsonr
from scipy.spatial.distance import squareform, pdist
import random
import copy

N_INDS_KEEP = 500
N_CLUSTERS = 100
N_CLUSTER_SAMPLE = 2000
cluster_batch_sz = 200

conv_block_cuda = cm.conv
F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 5211 # batch size
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
inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)
inds_keep_cluster = np.random.randint(3*32*32 * 3*32*32, size=N_CLUSTER_SAMPLE)

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500

sigma31 = np.zeros((N_C, N_INDS_KEEP), dtype='single')
sigma11 = np.zeros((N_INDS_KEEP, N_INDS_KEEP), dtype='single')

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
for batch in [6]:#range(1,7):
	t_start = time.time()
	z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))

	################### compute train err
	# load imgs
	x = z['data'] - imgs_mean
	x = x.reshape((3, 32, 32, 10000))
	x = x[:,:,:,:N_IMGS]
	px = x.reshape((3*32*32, N_IMGS))

	labels = np.asarray(z['labels'])[:N_IMGS].astype(int)
	
	if True:
		# initial clusters
		cluster_counts = np.ones(N_CLUSTERS)
		pixel_dists = np.einsum(px[:,:N_CLUSTERS],[0,1],px[:,:N_CLUSTERS],[2,1],[1,0,2]).reshape((N_CLUSTERS, 3*32*32 * 3*32*32))
		pixel_dists_crop = pixel_dists[:,inds_keep_cluster]
		
		img_clusters = np.zeros(N_IMGS,dtype='int')
		
		# group next images into clusters
		for cluster_step in range(N_CLUSTERS, N_IMGS-cluster_batch_sz, cluster_batch_sz):
			t_start = time.time()
			img_pixel_dists = np.einsum(px[:,cluster_step:cluster_step+cluster_batch_sz],[0,1],px[:,cluster_step:cluster_step+cluster_batch_sz],[2,1],[1,0,2]).reshape((cluster_batch_sz, 3*32*32 * 3*32*32))
			img_pixel_dists_crop = img_pixel_dists[:,inds_keep_cluster]
			c_mat = np.zeros(N_CLUSTERS)
			for img in range(cluster_batch_sz):
				for img2 in range(N_CLUSTERS):
					c_mat[img2] = pearsonr(img_pixel_dists_crop[img], pixel_dists_crop[img2])[0]
				img_clusters[cluster_step+img] = np.argmax(c_mat)
				pixel_dists[np.argmax(c_mat)] += img_pixel_dists[img]
				cluster_counts[np.argmax(c_mat)] += 1
			print cluster_step, time.time() - t_start
		pixel_dists /= cluster_counts[:,np.newaxis]
		pixel_dists = pixel_dists.reshape((N_CLUSTERS, 3*32*32, 3*32*32))
		savemat('/home/darren/pixel_dists_clustered_temp.mat',{'pixel_dists':pixel_dists, 'img_clusters':img_clusters})
	else:
		z = loadmat('/home/darren/pixel_dists_clustered_temp.mat')
		pixel_dists = z['pixel_dists'].reshape((N_CLUSTERS, 3*32*32, 3*32*32))
		img_clusters = z['img_clusters']
	
	
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
	patches, channels, x, y = patch_inds_addresses(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, imgs_pad, N_C, inds_keep, warn=False)
	print time.time() - t_patch
	
	x[x>=32] = 31
	y[y>=32] = 31
	inds = np.ravel_multi_index((np.int64(channels),np.int64(x),np.int64(y)),dims=(3,32,32))
	sigma11 = np.zeros((N_INDS_KEEP,N_INDS_KEEP))
	t_start = time.time()
	for i in range(N_INDS_KEEP):
		if i%100 == 0:
			print i, time.time() - t_start
			t_start = time.time()
		for j in range(i,N_INDS_KEEP):
			sigma11[i,j] = sigma11[j,i] = np.mean(pixel_dists[img_clusters, inds[:N_IMGS,i],inds[:N_IMGS,j]])
	
	for cat in range(N_C):
		sigma31[cat] += patches[labels == cat].sum(0) / N_IMGS
	print batch, time.time() - t_start


savemat('/home/darren/sigmas_train_test_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat',{'sigma11':sigma11, 'sigma31':sigma31, 'patches':patches,'labels':labels, 'inds_keep':inds_keep,'x':x,'y':y,'channels':channels})
