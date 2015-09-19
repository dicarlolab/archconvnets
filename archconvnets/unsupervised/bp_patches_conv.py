import time
import numpy as np
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import *
from archconvnets.unsupervised.cudnn_module.cudnn_module import *
from scipy.io import savemat, loadmat
from scipy.stats import zscore
import random
import gnumpy as gpu
import copy

#kernprof -l bp_patches_conv.py
#python -m line_profiler bp_patches_conv.py.lprof  > p
#@profile
#def sf():
TEST_FREQ = 50
N_TEST_SET = 500
N_TRAIN = 300
TOP_N = 1

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 100 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34#70#75# # input image size (px)
img_train_offset = 0
PAD = 2

N = 4
n1 = N # L1 filters
n2 = N
n3 = N

s1 = 5
s2 = 5
s3 = 3

N_C = 100 # number of categories

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

F1_init = copy.deepcopy(F1)
F2_init = copy.deepcopy(F2)
F3_init = copy.deepcopy(F3)
FL_init = copy.deepcopy(FL)

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
batch = 1
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))

################### compute test switches
# load imgs
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))[:,:,:,:N_TEST_SET]

labels = np.squeeze(np.asarray(z['labels']).astype(int))[:N_TEST_SET]
Y_test = np.zeros((10, len(labels)))
Y_test[labels, range(len(labels))] = 1

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_TEST_SET),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

# forward pass
t_forward_start = time.time()
conv_output1 = conv(F1, imgs_pad)
max_output1t, output_switches1_x_test, output_switches1_y_test = max_pool_locs(np.single(conv_output1),warn=False)

max_output1 = np.zeros((N_TEST_SET, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv(F2, max_output1)
max_output2t, output_switches2_x_test, output_switches2_y_test = max_pool_locs(np.single(conv_output2), PAD=2,warn=False)

max_output2 = np.zeros((N_TEST_SET, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv(F3, max_output2)
max_output3, output_switches3_x_test, output_switches3_y_test = max_pool_locs(np.single(conv_output3), PAD=2,warn=False)

output_switches2_x_test -= PAD
output_switches2_y_test -= PAD

output_switches3_x_test -= PAD
output_switches3_y_test -= PAD


class_test = []
EPS = 1e-4
step_total = 0
epoch = 0
while True:
	epoch += 1
	for batch in range(1,6):
		z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
		z['data'] = z['data'] - imgs_mean
		z['data'] = z['data'].reshape((3, 32, 32, 10000))
		for step in range(100):
			if (step % TEST_FREQ) == 0:
				t_patch = time.time()
				m = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(1))
				########################################################################################## test
				x = m['data'] - imgs_mean
				x = x.reshape((3, 32, 32, 10000))[:,:,:,:N_TEST_SET]
				
				imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_TEST_SET),dtype='single')
				imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
				imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

				conv_output1 = conv(F1, imgs_pad)
				max_output1t = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output1[:,np.newaxis])), output_switches1_x_test, output_switches1_y_test)
				max_output1 = np.zeros((N_TEST_SET, n1, max_output_sz1, max_output_sz1),dtype='single')
				max_output1[:,:, PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = np.squeeze(max_output1t)

				conv_output2 = conv(F2, max_output1)
				max_output2t = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output2[:,np.newaxis])), output_switches2_x_test, output_switches2_y_test)
				max_output2 = np.zeros((N_TEST_SET, n2, max_output_sz2, max_output_sz2),dtype='single')
				max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = np.squeeze(max_output2t)

				conv_output3 = conv(F3, max_output2)
				max_output3 = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output3[:,np.newaxis])), output_switches3_x_test, output_switches3_y_test)
				pred = zscore(max_output3.reshape((N_TEST_SET,n3*max_output_sz3**2)),axis=1)
				
				pred_train = pred[:N_TRAIN]
				pred = pred[N_TRAIN:]
				
				test_corrs = np.dot(pred, pred_train.T)
				hit = 0
				for test_img in range(N_TEST_SET-N_TRAIN):
					hit += np.max(labels[N_TRAIN + test_img] == labels[np.argsort(-test_corrs[test_img])[:TOP_N]])
				
				class_test.append(1 - hit/np.single(N_TEST_SET-N_TRAIN))

			########################################################################################## uns
			s_batch = step % 100
			x = z['data'][:,:,:,s_batch*100:(s_batch+1)*100]

			imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
			imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
			imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

			conv_output1 = conv(F1_init, imgs_pad)
			max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(np.single(conv_output1),warn=False)

			max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
			max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

			conv_output2 = conv(F2_init, max_output1)
			max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(np.single(conv_output2), PAD=2,warn=False)

			max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
			max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

			conv_output3 = conv(F3_init, max_output2)
			max_output3, output_switches3_x, output_switches3_y = max_pool_locs(np.single(conv_output3), PAD=2,warn=False)

			output_switches2_x -= PAD
			output_switches2_y -= PAD

			output_switches3_x -= PAD
			output_switches3_y -= PAD

			#### current filters:
			
			conv_output1 = conv(F1, imgs_pad)
			max_output1t, pool1_patches = max_pool_locs_alt_patches(conv_output1, output_switches1_x, output_switches1_y, imgs_pad, s1)
			max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
			max_output1[:,:, PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = np.squeeze(max_output1t)

			conv_output2 = conv(F2, max_output1)
			max_output2t, pool2_patches = max_pool_locs_alt_patches(conv_output2, output_switches2_x, output_switches2_y, max_output1, s2)
			max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
			max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = np.squeeze(max_output2t)

			conv_output3 = conv(F3, max_output2)
			max_output3, pool3_patches = max_pool_locs_alt_patches(conv_output3, output_switches3_x, output_switches3_y, max_output2, s3)
			pred = np.einsum(np.squeeze(max_output3), [0,1,2,3], FL, [4,1,2,3], [4,0]) # N_C x N_IMGS
			pred_ravel = pred.ravel()
			
			########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_
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
				
			FLr = FL.reshape((N_C, n3*max_output_sz3**2))
			
			FLr_g = gpu.garray(FLr)
			max_output3t_accum_g = gpu.garray(max_output3t_accum.transpose((0,1,2,3,5,4)))
			pred_deriv_L1 = FLr_g.dot(max_output3t_accum_g).as_numpy_array()
			
			
			#pred_deriv_L1 = np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4))) # sum across f3,z1,z2
			
			pred_deriv_L1 = pred_deriv_L1.reshape((N_C*N_IMGS, n1, 3, s1, s1)).transpose((1,2,3,0,4))
			pred_pred_deriv_L1 = np.dot(pred_ravel, pred_deriv_L1)
			
			########## F2 deriv wrt f2_, a2_x_, a2_y_, f1_
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
				
			max_output3t_accum_g = gpu.garray(max_output3t_accum.transpose((0,1,2,3,5,4)))
			pred_deriv_L2 = FLr_g.dot(max_output3t_accum_g).as_numpy_array()
			
			#pred_deriv_L2 = np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4)))
			
			pred_deriv_L2 = pred_deriv_L2.reshape((N_C*N_IMGS, n2, n1, s2, s2)).transpose((1,2,3,0,4))
			pred_pred_deriv_L2 = np.dot(pred_ravel, pred_deriv_L2)

			########## F3 deriv wrt f3_, a3_x_, a3_y_, f2_
			pred_pred_deriv_L3 = np.zeros_like(F3)
			for a3_x_ in range(s3):
				for a3_y_ in range(s3):
					for f2_ in range(n2):
						pool3_deriv = pool3_patches[:,f2_,a3_x_,a3_y_]
						for f3_ in range(n3):
							pred_deriv = np.dot(FL[:,f3_].reshape((N_C, max_output_sz3**2)), pool3_deriv[:,f3_].reshape((N_IMGS, max_output_sz3**2)).T).ravel()
							
							pred_pred_deriv_L3[f3_, f2_, a3_x_, a3_y_] = np.dot(pred_deriv, pred_ravel)

			########## FL deriv wrt cat_, f3_, z1_, z2_
			pred_pred_deriv_FL = np.einsum(pred,[4,0],max_output3,range(4),[4,1,2,3])
			
			########################################################################################## sup
			x = z['data'][:,:,:,:N_C]
			
			imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_C),dtype='single')
			imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
			imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))
			
			conv_output1 = conv(F1_init, imgs_pad)
			max_output1t, output_switches1_x_sup, output_switches1_y_sup = max_pool_locs(np.single(conv_output1),warn=False)

			max_output1 = np.zeros((N_C, n1, max_output_sz1, max_output_sz1),dtype='single')
			max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

			conv_output2 = conv(F2_init, max_output1)
			max_output2t, output_switches2_x_sup, output_switches2_y_sup = max_pool_locs(np.single(conv_output2), PAD=2,warn=False)

			max_output2 = np.zeros((N_C, n2, max_output_sz2, max_output_sz2),dtype='single')
			max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

			conv_output3 = conv(F3_init, max_output2)
			max_output3, output_switches3_x_sup, output_switches3_y_sup = max_pool_locs(np.single(conv_output3), PAD=2,warn=False)

			output_switches2_x_sup -= PAD
			output_switches2_y_sup -= PAD

			output_switches3_x_sup -= PAD
			output_switches3_y_sup -= PAD
			
			#### current filters:
			
			conv_output1 = conv(F1, imgs_pad)
			max_output1t, pool1_patches = max_pool_locs_alt_patches(conv_output1, output_switches1_x_sup, output_switches1_y_sup, imgs_pad, s1)
			max_output1 = np.zeros((N_C, n1, max_output_sz1, max_output_sz1),dtype='single')
			max_output1[:,:, PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = np.squeeze(max_output1t)

			conv_output2 = conv(F2, max_output1)
			max_output2t, pool2_patches = max_pool_locs_alt_patches(conv_output2, output_switches2_x_sup, output_switches2_y_sup, max_output1, s2)
			max_output2 = np.zeros((N_C, n2, max_output_sz2, max_output_sz2),dtype='single')
			max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = np.squeeze(max_output2t)

			conv_output3 = conv(F3, max_output2)
			max_output3, pool3_patches = max_pool_locs_alt_patches(conv_output3, output_switches3_x_sup, output_switches3_y_sup, max_output2, s3)
			
			########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_
			pool1_derivt = pool1_patches.reshape((N_C*3*s1*s1, n1, max_output_sz1-2*PAD, max_output_sz1-2*PAD))
			pool1_deriv = np.zeros((N_C*3*s1*s1, n1, max_output_sz1, max_output_sz1),dtype='single')
			pool1_deriv[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = pool1_derivt

			pool1_deriv = np.ascontiguousarray(pool1_deriv.transpose((1,0,2,3))[:,:,np.newaxis])
			F2c = np.ascontiguousarray(F2.transpose((1,0,2,3))[:,:,np.newaxis])

			max_output3t_accum = np.zeros((N_C, n1, 3, s1, s1, n3*max_output_sz3**2),dtype='single')
			for f1_ in range(n1):
				conv_output2_deriv = conv(F2c[f1_], pool1_deriv[f1_])
				conv_output2_deriv = conv_output2_deriv.reshape((N_C, 3*s1*s1, n2, output_sz2, output_sz2))
				
				max_output2t = max_pool_locs_alt(conv_output2_deriv, output_switches2_x_sup, output_switches2_y_sup)
				max_output2 = np.zeros((N_C, 3*s1*s1, n2, max_output_sz2, max_output_sz2),dtype='single')
				max_output2[:,:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t
				max_output2 = max_output2.reshape((N_C*3*s1*s1, n2, max_output_sz2, max_output_sz2))
				
				conv_output3_deriv = conv(F3, max_output2)
				conv_output3_deriv = conv_output3_deriv.reshape((N_C, 3*s1*s1, n3, output_sz3, output_sz3))
				
				max_output3t = max_pool_locs_alt(conv_output3_deriv, output_switches3_x_sup, output_switches3_y_sup)
				max_output3t_accum[:,f1_] = max_output3t.reshape((N_C, 3, s1, s1, n3*max_output_sz3**2))
				
			max_output3t_accum_g = gpu.garray(max_output3t_accum.transpose((0,1,2,3,5,4)))
			pred_deriv_L1_sup = FLr_g.dot(max_output3t_accum_g).as_numpy_array()
			
			#pred_deriv_L1_sup = np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4))) # sum across f3,z1,z2
			
			########## F2 deriv wrt f2_, a2_x_, a2_y_, f1_
			pool2_derivt = pool2_patches.reshape((N_C*n1*s2*s2, n2, max_output_sz2-2*PAD, max_output_sz2-2*PAD))
			pool2_deriv = np.zeros((N_C*n1*s2*s2, n2, max_output_sz2, max_output_sz2),dtype='single')
			pool2_deriv[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = pool2_derivt

			pool2_deriv = np.ascontiguousarray(pool2_deriv.transpose((1,0,2,3))[:,:,np.newaxis])
			F3c = np.ascontiguousarray(F3.transpose((1,0,2,3))[:,:,np.newaxis])

			max_output3t_accum = np.zeros((N_C, n2, n1, s2, s2, n3*max_output_sz3**2),dtype='single')
			for f2_ in range(n2):
				conv_output3_deriv = conv(F3c[f2_], pool2_deriv[f2_])
				conv_output3_deriv = conv_output3_deriv.reshape((N_C, n1*s2*s2, n3, output_sz3, output_sz3))
				
				max_output3t = max_pool_locs_alt(conv_output3_deriv, output_switches3_x_sup, output_switches3_y_sup)
				max_output3t_accum[:,f2_] = max_output3t.reshape((N_C, n1, s2, s2, n3*max_output_sz3**2))
				
			max_output3t_accum_g = gpu.garray(max_output3t_accum.transpose((0,1,2,3,5,4)))
			pred_deriv_L2_sup = FLr_g.dot(max_output3t_accum_g).as_numpy_array()
			#pred_deriv_L2_sup = np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4)))
			
			########## F3 deriv wrt f3_, a3_x_, a3_y_, f2_
			pred_deriv_L3_sup = np.zeros((N_C, N_C, n3, n2, s3, s3), dtype='single')
			for a3_x_ in range(s3):
				for a3_y_ in range(s3):
					for f2_ in range(n2):
						pool3_deriv = pool3_patches[:,f2_,a3_x_,a3_y_]
						for f3_ in range(n3):
							pred_deriv_L3_sup[:,:,f3_,f2_,a3_x_,a3_y_] = np.dot(FL[:,f3_].reshape((N_C, max_output_sz3**2)), pool3_deriv[:,f3_].reshape((N_C, max_output_sz3**2)).T)

			########## FL deriv wrt cat_, f3_, z1_, z2_
			pred_deriv_FL_sup = np.squeeze(max_output3)
			
			
			grad_F1 = (pred_pred_deriv_L1 / (N_C * N_IMGS)) - pred_deriv_L1_sup.mean(0).mean(0)
			grad_F2 = (pred_pred_deriv_L2 / (N_C * N_IMGS)) - pred_deriv_L2_sup.mean(0).mean(0)
			grad_F3 = (pred_pred_deriv_L3 / (N_C * N_IMGS)) - pred_deriv_L3_sup.mean(0).mean(0)
			grad_FL = (pred_pred_deriv_FL / (N_C * N_IMGS)) - pred_deriv_FL_sup
			
			F1 -= EPS*grad_F1
			F2 -= EPS*grad_F2
			F3 -= EPS*grad_F3
			FL -= EPS*grad_FL
			
			step_total += 1
			if (step_total % TEST_FREQ) == 0:
				savemat('/home/darren/F1.mat', {'F1': F1, 'step':step_total, 'class_test':class_test})
				print epoch, batch, step, class_test[-1], time.time() - t_patch

sf()