from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
import numexpr as ne
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import max_pool_locs
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu

#kernprof -l bp_cudnn.py
#python -m line_profiler bp_cudnn.py.lprof  > p
#@profile
#def sf():
FBUFF_F1 = 0
FBUFF_F2 = 1
FBUFF_F3 = 2

FBUFF_F1_init = 3
FBUFF_F2_init = 4
FBUFF_F3_init = 5
FBUFF_F3_GRAD_L2 = 6
FBUFF_F2_GRAD_L1 = 7

IBUFF_TRAIN_IMGS = 0
IBUFF_TEST_IMGS = 1
IBUFF_POOL2 = 4
IBUFF_F2_GRAD_L1 = 7
IBUFF_F3_GRAD_L2 = 8

CBUFF_F1_TRAIN_IMGS = 1
CBUFF_F1_init_TRAIN_IMGS = 2

CBUFF_F1_TEST_IMGS = 3
CBUFF_F1_init_TEST_IMGS = 4

CBUFF_F3_GRAD_F1 = 5

CBUFF_F3_GRAD_L2 = 6
CBUFF_F2_GRAD_L1 = 7

conv_block_cuda = conv

filename = '/home/darren/cifar_sigma_eps_test.mat'

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

EPS = 1e-3#1e-2
eps_F1 = EPS
eps_F2 = EPS
eps_F3 = EPS
eps_FL = EPS

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 128 # batch size
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

if False:
	x = loadmat(filename)
	F1 = x['F1']
	F2 = x['F2']
	F3 = x['F3']
	FL = x['FL']
	class_err = x['class_err'].tolist()
	class_err_test = x['class_err_test'].tolist()
	err = x['err'].tolist()
	err_test = x['err_test'].tolist()
	epoch_err = x['epoch_err'].tolist()
	
	np.random.seed(623)
	F1_init = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
	F2_init = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
	F3_init = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
	FL_init = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))
else:
	np.random.seed(6666)
	F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
	F1_init = copy.deepcopy(F1)
	F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
	F2_init = copy.deepcopy(F2)
	F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
	F3_init = copy.deepcopy(F3)
	FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))
	FL_init = copy.deepcopy(FL)
	err = []
	class_err = []
	err_test = []
	class_err_test = []
	epoch_err = []

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

grad_L1_s = 0
grad_L2_s = 0
grad_L3_s = 0
grad_FL_s = 0

grad_L1_uns = 0
grad_L2_uns = 0
grad_L3_uns = 0
grad_FL_uns = 0


set_filter_buffer(FBUFF_F1_init, F1_init)
set_filter_buffer(FBUFF_F2_init, F2_init)
set_filter_buffer(FBUFF_F3_init, F3_init)

##################
# load test imgs into buffers
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_TEST_IMGS]

l = np.zeros((N_TEST_IMGS, N_C),dtype='int')
l[np.arange(N_TEST_IMGS),np.asarray(z['labels'])[:N_TEST_IMGS].astype(int)] = 1
Y_test = np.double(l.T)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_TEST_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP]
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

set_img_buffer(IBUFF_TEST_IMGS, imgs_pad)
set_conv_buffer(CBUFF_F1_init_TEST_IMGS, FBUFF_F1_init, IBUFF_TEST_IMGS)

# forward pass init filters on test imgs
conv_output1 = conv_from_buffers(CBUFF_F1_init_TEST_IMGS)
max_output1t, output_switches1_x_init, output_switches1_y_init = max_pool_locs(conv_output1)
max_output1 = np.zeros((N_TEST_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv_block_cuda(F2_init, max_output1)
max_output2t, output_switches2_x_init, output_switches2_y_init = max_pool_locs(conv_output2)
max_output2 = np.zeros((N_TEST_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv_block_cuda(F3_init, max_output2)
max_output3, output_switches3_x_init, output_switches3_y_init = max_pool_locs(conv_output3)

for iter in range(np.int(1e7)):
	epoch_err_t = 0
	for batch in range(1,6):
		z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
		x = z['data'] - imgs_mean
		x = x.reshape((3, 32, 32, 10000))
		
		for step in range(np.int((10000)/N_IMGS)):
			t_total = time.time()
			
			set_filter_buffer(FBUFF_F1, F1)
			set_filter_buffer(FBUFF_F2, F2)
			set_filter_buffer(FBUFF_F3, F3)
			
			set_conv_buffer(CBUFF_F1_TEST_IMGS, FBUFF_F1, IBUFF_TEST_IMGS)
			
			FLr = FL.reshape((N_C, n3*max_output_sz3**2))
			FLrg = gpu.garray(FLr)
			
			########################## compute test err
			t_test_forward_start = time.time()
			
			# forward pass current filters
			t_test_forward_start = time.time()
			conv_output1 = conv_from_buffers(CBUFF_F1_TEST_IMGS)
			max_output1t = max_pool_locs_alt(np.ascontiguousarray(conv_output1[:,np.newaxis]), output_switches1_x_init, output_switches1_y_init)
			max_output1 = np.zeros((N_TEST_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
			max_output1[:,:, PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = np.squeeze(max_output1t)

			conv_output2 = conv_block_cuda(F2, max_output1)
			max_output2t = max_pool_locs_alt(np.ascontiguousarray(conv_output2[:,np.newaxis]), output_switches2_x_init, output_switches2_y_init)
			max_output2 = np.zeros((N_TEST_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
			max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = np.squeeze(max_output2t)

			conv_output3 = conv_block_cuda(F3, max_output2)
			max_output3 = max_pool_locs_alt(np.ascontiguousarray(conv_output3[:,np.newaxis]), output_switches3_x_init, output_switches3_y_init)

			pred = np.dot(FLr, max_output3.reshape((N_TEST_IMGS, n3*max_output_sz3**2)).T)
			err_test.append(np.sum((pred - Y_test)**2)/N_TEST_IMGS)
			class_err_test.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y_test, axis=0)))/N_TEST_IMGS)
			
			t_test_forward_start = time.time() - t_test_forward_start
			t_forward_start = time.time()
			
			################### compute train err
			# load imgs
			t_grad_start = time.time()
			t_forward_start = time.time()
			err.append(0)
			class_err.append(0)

			l = np.zeros((N_IMGS, N_C),dtype='int')
			img_cats = np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)
			l[np.arange(N_IMGS),np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)] = 1
			Y = np.double(l.T)

			imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
			offset_x = np.random.randint(IMG_SZ - IMG_SZ_CROP)
			offset_y = np.random.randint(IMG_SZ - IMG_SZ_CROP)
			imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,offset_x:offset_x+IMG_SZ_CROP,offset_y:offset_y+IMG_SZ_CROP,step*N_IMGS:(step+1)*N_IMGS]
			imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

			t_forward_start = time.time()
			
			set_img_buffer(IBUFF_TRAIN_IMGS, imgs_pad)
			set_conv_buffer(CBUFF_F1_TRAIN_IMGS, FBUFF_F1, IBUFF_TRAIN_IMGS)
			set_conv_buffer(CBUFF_F1_init_TRAIN_IMGS, FBUFF_F1_init, IBUFF_TRAIN_IMGS)
			
			# forward pass init filters
			conv_output1 = conv_from_buffers(CBUFF_F1_init_TRAIN_IMGS) #
			max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)
			max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
			max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

			conv_output2 = conv_block_cuda(F2_init, max_output1)
			max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)
			max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
			max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

			conv_output3 = conv_block_cuda(F3_init, max_output2)
			max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)
	
			# forward pass current filters
			conv_output1 = conv_from_buffers(CBUFF_F1_TRAIN_IMGS) #
			max_output1t, pool1_patches = max_pool_locs_alt_patches(conv_output1, output_switches1_x, output_switches1_y, imgs_pad, s1)
			max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
			max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

			conv_output2 = conv_block_cuda(F2, max_output1)
			max_output2t, pool2_patches = max_pool_locs_alt_patches(conv_output2, output_switches2_x, output_switches2_y, max_output1, s2)
			max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
			max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

			conv_output3 = conv_block_cuda(F3, max_output2)
			max_output3, pool3_patches = max_pool_locs_alt_patches(conv_output3, output_switches3_x, output_switches3_y, max_output2, s3)
			max_output3 = np.squeeze(max_output3)
			
			pred = np.dot(FLr, max_output3.reshape((N_IMGS, n3*max_output_sz3**2)).T)
			err.append(np.sum((pred - Y)**2)/N_IMGS)
			class_err.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y,axis=0)))/N_IMGS)
			
			#random.shuffle(img_cats)
			pred[img_cats, range(N_IMGS)] -= 1 ############ backprop supervised term
			#pred[img_cats[:20], range(20)] -= 10 ############ backprop supervised term (tenth supervised)
			pred_ravel = pred.ravel()
			err.append(0)
			class_err.append(0)
			
			t_forward_start = time.time() - t_forward_start
			t_grad_start = time.time()
			
			
			########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_
			grad = np.zeros_like(F1)
			
			# ravel together all the patches to reduce the needed convolution function calls
			pool1_derivt = pool1_patches.reshape((N_IMGS*3*s1*s1, n1, max_output_sz1-2*PAD, max_output_sz1-2*PAD))
			pool1_deriv = np.zeros((N_IMGS*3*s1*s1, n1, max_output_sz1, max_output_sz1),dtype='single')
			pool1_deriv[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = pool1_derivt
			
			pool1_deriv = np.ascontiguousarray(pool1_deriv.transpose((1,0,2,3))[:,:,np.newaxis])
			F2c = np.ascontiguousarray(F2.transpose((1,0,2,3))[:,:,np.newaxis])
			
			max_output3t_accum = np.zeros((N_IMGS, n1, 3, s1, s1, n3*max_output_sz3**2),dtype='single')
			for f1_ in range(n1):
				set_filter_buffer(FBUFF_F2_GRAD_L1, F2c[f1_])
				set_img_buffer(IBUFF_F2_GRAD_L1, pool1_deriv[f1_])
				set_conv_buffer(CBUFF_F2_GRAD_L1, FBUFF_F2_GRAD_L1, IBUFF_F2_GRAD_L1)
				
				conv_output2_deriv = conv_from_buffers(CBUFF_F2_GRAD_L1)
				conv_output2_deriv = conv_output2_deriv.reshape((N_IMGS, 3*s1*s1, n2, output_sz2, output_sz2))
				
				max_output2t = max_pool_locs_alt(conv_output2_deriv, output_switches2_x, output_switches2_y)
				max_output2 = np.zeros((N_IMGS, 3*s1*s1, n2, max_output_sz2, max_output_sz2),dtype='single')
				max_output2[:,:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t
				max_output2 = max_output2.reshape((N_IMGS*3*s1*s1, n2, max_output_sz2, max_output_sz2))
				
				set_img_buffer(IBUFF_POOL2, max_output2);
				set_conv_buffer(CBUFF_F3_GRAD_F1, FBUFF_F3, IBUFF_POOL2)
				conv_output3_deriv = conv_from_buffers(CBUFF_F3_GRAD_F1)
				conv_output3_deriv = conv_output3_deriv.reshape((N_IMGS, 3*s1*s1, n3, output_sz3, output_sz3))
				
				max_output3t = max_pool_locs_alt(conv_output3_deriv, output_switches3_x, output_switches3_y)
				max_output3t_accum[:,f1_] = max_output3t.reshape((N_IMGS, 3, s1, s1, n3*max_output_sz3**2))
				
			#mtg = gpu.garray(max_output3t_accum.transpose((0,1,2,3,5,4)))
			#t=FLrg.dot(mtg).as_numpy_array()
			t=np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4)))

			t = t.reshape((N_C*N_IMGS, n1, 3, s1, s1)).transpose((1,2,3,0,4))
			grad_L1_uns = np.dot(pred_ravel, t) / N_IMGS
			
			########## F2 deriv wrt f2_, a2_x_, a2_y_, f1_
			grad = np.zeros_like(F2)
			
			# ravel together all the patches to reduce the needed convolution function calls
			pool2_derivt = pool2_patches.reshape((N_IMGS*n1*s2*s2, n2, max_output_sz2-2*PAD, max_output_sz2-2*PAD))
			pool2_deriv = np.zeros((N_IMGS*n1*s2*s2, n2, max_output_sz2, max_output_sz2),dtype='single')
			pool2_deriv[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = pool2_derivt
			
			pool2_deriv = np.ascontiguousarray(pool2_deriv.transpose((1,0,2,3))[:,:,np.newaxis])
			F3c = np.ascontiguousarray(F3.transpose((1,0,2,3))[:,:,np.newaxis])
			
			max_output3t_accum = np.zeros((N_IMGS, n2, n1, s2, s2, n3*max_output_sz3**2),dtype='single')
			for f2_ in range(n2):
				set_filter_buffer(FBUFF_F3_GRAD_L2, F3c[f2_])
				set_img_buffer(IBUFF_F3_GRAD_L2, pool2_deriv[f2_])
				set_conv_buffer(CBUFF_F3_GRAD_L2, FBUFF_F3_GRAD_L2, IBUFF_F3_GRAD_L2)
				
				conv_output3_deriv = conv_from_buffers(CBUFF_F3_GRAD_L2)
				conv_output3_deriv = conv_output3_deriv.reshape((N_IMGS, n1*s2*s2, n3, output_sz3, output_sz3))
				
				max_output3t = max_pool_locs_alt(conv_output3_deriv, output_switches3_x, output_switches3_y)
				max_output3t_accum[:,f2_] = max_output3t.reshape((N_IMGS, n1, s2, s2, n3*max_output_sz3**2))
				
			#mtg = gpu.garray(max_output3t_accum.transpose((0,1,2,3,5,4)))
			#t=FLrg.dot(mtg).as_numpy_array()
			t=np.dot(FLr, max_output3t_accum.transpose((0,1,2,3,5,4)))
			
			t = t.reshape((N_C*N_IMGS, n2, n1, s2, s2)).transpose((1,2,3,0,4))
			grad_L2_uns = np.dot(pred_ravel, t) / N_IMGS

			########## F3 deriv wrt f3_, a3_x_, a3_y_, f2_
			grad = np.zeros_like(F3)
			for a3_x_ in range(s3):
				for a3_y_ in range(s3):
					for f2_ in range(n2):
						pool3_deriv = pool3_patches[:,f2_,a3_x_,a3_y_]
						for f3_ in range(n3):
							pred_deriv = np.dot(FL[:,f3_].reshape((N_C, max_output_sz3**2)), pool3_deriv[:,f3_].reshape((N_IMGS, max_output_sz3**2)).T).ravel()
							
							grad[f3_, f2_, a3_x_, a3_y_] = np.dot(pred_deriv, pred_ravel)
							
			grad_L3_uns = grad / N_IMGS
			
			########## FL deriv wrt cat_, f3_, z1_, z2_
			grad_FL_uns = np.tile((pred[:,:,np.newaxis,np.newaxis,np.newaxis]*max_output3[np.newaxis]).sum(0).sum(0)[np.newaxis], (N_C,1,1,1)) / N_IMGS
			
			##########
			# weight updates
			v_i1_L1 = -eps_F1 * (grad_L1_uns + grad_L1_s)
			v_i1_L2 = -eps_F2 * (grad_L2_uns + grad_L2_s)
			v_i1_L3 = -eps_F3 * (grad_L3_uns + grad_L3_s)
			v_i1_FL = -eps_FL * (grad_FL_uns + grad_FL_s)
			
			################
			# diagnostics
			grad_L1_uns *= eps_F1
			grad_L2_uns *= eps_F2
			grad_L3_uns *= eps_F3
			grad_FL_uns *= eps_FL
			
			F1 += v_i1_L1
			F2 += v_i1_L2
			F3 += v_i1_L3
			FL += v_i1_FL
			
			#######################################
			
			print iter, batch, step, err_test[-1], class_err_test[-1],  time.time() - t_total, filename
			print '                        F1', np.mean(np.abs(v_i1_L1))/np.mean(np.abs(F1)), 'F2', np.mean(np.abs(v_i1_L2))/np.mean(np.abs(F2)), 'F3', np.mean(np.abs(v_i1_L3))/np.mean(np.abs(F3)), 'FL', np.mean(np.abs(v_i1_FL))/np.mean(np.abs(FL))
			print '                        F1', np.mean(np.abs(F1)), 'F2', np.mean(np.abs(F2)), 'F3', np.mean(np.abs(F3)), 'FL', np.mean(np.abs(FL)), ' m'
			savemat(filename, {'F1': F1, 'F2': F2, 'F3':F3, 'FL': FL, 'eps_FL': eps_FL, 'eps_F3': eps_F3, 'eps_F2': eps_F2, 'step': step, 'eps_F1': eps_F1, 'N_IMGS': N_IMGS, 'N_TEST_IMGS': N_TEST_IMGS,'err_test':err_test,'err':err,'class_err':class_err,'class_err_test':class_err_test,'epoch_err':epoch_err})
			epoch_err_t += err[-1]
	epoch_err.append(epoch_err_t)
	print '------------ epoch err ----------'
	print epoch_err_t
sf()
