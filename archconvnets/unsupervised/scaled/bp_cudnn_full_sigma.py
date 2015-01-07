from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
import numexpr as ne
from archconvnets.unsupervised.pool_inds import max_pool_locs
#from archconvnets.unsupervised.pool_alt_inds_opt import max_pool_locs_alt
#from archconvnets.unsupervised.pool_alt_inds_opt_patches import max_pool_locs_alt_patches
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu

#kernprof -l bp_cudnn_full_sigma.py
#python -m line_profiler bp_cudnn_full_sigma.py.lprof  > p
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

filename = '/home/darren/cifar_8N_full_eps5e6_all_layers.mat'

S_SCALE = 20#1e-2
WD = 0#1
MOMENTUM = 0#0.9

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

EPS = 5e7#5e-3
eps_F1 = EPS
eps_F2 = EPS
eps_F3 = EPS
eps_FL = EPS

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 5000 # batch size
N_TEST_IMGS = N_IMGS #N_SIGMA_IMGS #128*2
N_SIGMA_IMGS = N_IMGS
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
F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500
FL = zscore(FL,axis=None)/500
imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

v_i_L1 = 0
v_i_L2 = 0
v_i_L3 = 0
v_i_FL = 0


y = loadmat('/home/darren/sigma31full_8N_10000imgs.mat')
sigma31 = y['sigma31']
sigma31 = sigma31.transpose((0,2,1,3,4,5,6,7,8,9,10,11,12))

# 10, n1, 3, s1, s1, n2, s2, s2, n3, s3, s3, sz, sz

#sigma31_LF = sigma31.mean(1).mean(1).mean(1).mean(1).mean(1).mean(1).mean(1).mean(2).mean(2)
#sigma31_LF = sigma31_LF.reshape((N_C, 1, 1, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))

#sigma31_LF = sigma31.mean(2).mean(2).mean(2).mean(2).mean(2).mean(2).mean(3).mean(3)
#sigma31_LF = sigma31_LF.reshape((N_C, n1, 1, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))

sigma31_LF = sigma31.mean(5).mean(5).mean(5).mean(6).mean(6)
sigma31_LF = sigma31_LF.reshape((N_C, n1, 3, s1, s1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))

sigma31_L3 = sigma31.mean(-1).mean(-1).mean(6).mean(6)
sigma31_L3 = sigma31_L3.reshape((N_C, n1, 3, s1, s1, n2, 1, 1, n3, s3, s3, 1, 1))

#sigma31_L3 = sigma31.mean(3).mean(3).mean(4).mean(4).mean(-1).mean(-1)
#sigma31_L3 = sigma31_L3.reshape((N_C, n1, 3, 1, 1, n2, 1, 1, n3, s3, s3, 1, 1))


#sigma31_L2 = sigma31.mean(-1).mean(-1).mean(-1).mean(-1).mean(-1).mean(2).mean(2).mean(2)
#sigma31_L2 = sigma31_L2.reshape((N_C, n1, 1, 1, 1, n2, s2, s2, 1, 1, 1, 1, 1))

sigma31_L2 = sigma31.mean(-1).mean(-1).mean(-1).mean(-1).mean(-1)
sigma31_L2 = sigma31_L2.reshape((N_C, n1, 3, s1, s1, n2, s2, s2, 1, 1, 1, 1, 1))


#sigma31_L1 = sigma31.mean(-1).mean(-1).mean(-1).mean(-1).mean(-1)
#sigma31_L1 = sigma31_L1.reshape((N_C, n1, 3, s1, s1, n2, s2, s2, 1, 1, 1, 1, 1))

sigma31_L1 = sigma31.mean(-1).mean(-1).mean(-1).mean(-1).mean(-1).mean(-1).mean(-1)
sigma31_L1 = sigma31_L1.reshape((N_C, n1, 3, s1, s1, n2, 1, 1, 1, 1, 1, 1, 1))

print 'sigma loaded'


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
		for step in range(np.int((10000)/N_IMGS)):
			#F1 = zscore(F1,axis=None)
			#F2 = zscore(F2,axis=None)
			#F3 = zscore(F3,axis=None)
			#FL = zscore(FL,axis=None)
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
			
			
			t_grad_start = time.time() - t_grad_start
			
			t_grad_s_start = time.time()
			
			FLt = FL.reshape((N_C, 1, 1, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))
			
			sigma_inds = [0,2,3,4,5,6,7,8,9,10,11,12,13]
			F_inds = [1,2,3,4,5,6,7,8,9,10,11,12,13]
			
			############################################## F1 deriv wrt f1_, a1_x_, a1_y_, channel_
			F32 = F2[np.newaxis,:,:,:,:,np.newaxis,np.newaxis] * F3[:,:,np.newaxis,np.newaxis,np.newaxis]
			# F32: n3, n2, n1, s2,s2, s3,s3
			F32 = F32.transpose((2,1,3,4,0,5,6))
			# F32: n1, n2, s2,s2, n3, s3,s3
			F32 = F32[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:,:,:,:,:,:,np.newaxis,np.newaxis]
			FL32 = FLt * F32
			
			sigma31_F1 = sigma31_L1 * F1.reshape((1, n1, 3, s1, s1,  1, 1, 1, 1, 1, 1, 1, 1))
			
			derivc = np.einsum(sigma31_L1, sigma_inds, FL32, F_inds, range(6))
			predc = np.einsum(sigma31_F1, sigma_inds, FL32, F_inds, range(6))
			grad_L1_s = (predc*derivc).sum(0).sum(0)
			grad_L1_s -=  np.einsum(derivc,[0,0,2,3,4,5], [2,3,4,5])
			
			
			############################################# F2 deriv wrt f2_, f1_, a2_x_, a2_y_
			F31 = np.tensordot(F1, F3, 0)
			F31 = F31.reshape((1,n1, 3, s1, s1, n2, 1,1, n3, s3, s3, 1, 1))
			FL31 = FLt * F31
			
			sigma31_F2 = sigma31_L2 * F2.transpose((1,0,2,3)).reshape((1, n1, 1, 1, 1, n2, s2, s2, 1, 1, 1, 1, 1))
			
			derivc = np.einsum(sigma31_L2, sigma_inds, FL31, F_inds, [0,1,6,2,7,8])
			predc = np.einsum(sigma31_F2, sigma_inds, FL31, F_inds, [0,1,6,2,7,8])
			grad_L2_s = (predc*derivc).sum(0).sum(0)
			grad_L2_s -=  np.einsum(derivc,[0,0,2,3,4,5], [2,3,4,5])
			
			
			############################################## F3 deriv wrt f3_, f2_, a3_x_, a3_y_
			F21 = F1[:,:,:,:,np.newaxis,np.newaxis,np.newaxis] * F2.transpose((1,0,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
			F21 = F21.reshape((1, n1, 3, s1, s1, n2, s2, s2, 1, 1, 1, 1, 1))
			FL21 = FLt * F21
			
			sigma31_F3 = sigma31_L3 * F3.transpose((1,0,2,3)).reshape((1, 1, 1, 1, 1, n2, 1, 1, n3, s3, s3, 1, 1))
			
			derivc = np.einsum(sigma31_L3, sigma_inds, FL21, F_inds, [0,1,9,6,10,11])
			predc = np.einsum(sigma31_F3, sigma_inds, FL21, F_inds, [0,1,9,6,10,11])
			grad_L3_s = (predc*derivc).sum(0).sum(0)
			grad_L3_s -=  np.einsum(derivc,[0,0,2,3,4,5], [2,3,4,5])
			
			
			####################################### FL deriv wrt cat_, f3_, z1_, z2_
			grad_FL_s = np.zeros_like(FL)
			
			F321 = F21 * F3.transpose((1,0,2,3)).reshape((1, 1, 1, 1, 1, n2, 1, 1, n3, s3, s3, 1, 1))
			
			sigma31_FL = sigma31_LF * FL.reshape((N_C, 1, 1, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))
			
			derivc = np.einsum(sigma31_LF, sigma_inds, F321, F_inds, [0,9,12,13])
			predc = np.einsum(sigma31_FL, sigma_inds, F321, F_inds, [0,9,12,13])
			grad_FL_s = derivc*predc - derivc
			

			
			##########
			# weight updates
			v_i1_L1 = -eps_F1 * (WD * F1 + grad_L1_uns + grad_L1_s) + MOMENTUM * v_i_L1
			v_i1_L2 = -eps_F2 * (WD * F2 + grad_L2_uns + grad_L2_s) + MOMENTUM * v_i_L2
			v_i1_L3 = -eps_F3 * (WD * F3 + grad_L3_uns + grad_L3_s) + MOMENTUM * v_i_L3
			v_i1_FL = -eps_FL * (WD * FL + grad_FL_uns + grad_FL_s) + MOMENTUM * v_i_FL
			
			F1 += v_i1_L1
			F2 += v_i1_L2
			F3 += v_i1_L3
			FL += v_i1_FL
			
			v_i_L1 = v_i1_L1
			v_i_L2 = v_i1_L2
			v_i_L3 = v_i1_L3
			v_i_FL = v_i1_FL
			
			
			#######################################
			
			print iter, batch, step, err_test[-1], class_err_test[-1],  time.time() - t_grad_s_start, time.time() - t_total, filename
			print '                        F1', np.mean(np.abs(v_i1_L1))/np.mean(np.abs(F1)), 'F2', np.mean(np.abs(v_i1_L2))/np.mean(np.abs(F2)), 'F3', np.mean(np.abs(v_i1_L3))/np.mean(np.abs(F3)), 'FL', np.mean(np.abs(v_i1_FL))/np.mean(np.abs(FL))
			#print '                        F1', np.mean(np.abs(grad_L1_uns))/np.mean(np.abs(F1)), 'F2', np.mean(np.abs(grad_L2_uns))/np.mean(np.abs(F2)), 'F3', np.mean(np.abs(grad_L3_uns))/np.mean(np.abs(F3)), 'FL', np.mean(np.abs(grad_FL_uns))/np.mean(np.abs(FL)), ' uns'
			#print '                        F1', np.mean(np.abs(grad_L1_s))/np.mean(np.abs(F1)), 'F2', np.mean(np.abs(grad_L2_s))/np.mean(np.abs(F2)), 'F3', np.mean(np.abs(grad_L3_s))/np.mean(np.abs(F3)), 'FL', np.mean(np.abs(grad_FL_s))/np.mean(np.abs(FL)), ' s'
			print '                        F1', np.mean(np.abs(F1)), 'F2', np.mean(np.abs(F2)), 'F3', np.mean(np.abs(F3)), 'FL', np.mean(np.abs(FL)), ' m'
			savemat(filename, {'F1': F1, 'F2': F2, 'F3':F3, 'FL': FL, 'eps_FL': eps_FL, 'eps_F3': eps_F3, 'eps_F2': eps_F2, 'step': step, 'eps_F1': eps_F1, 'N_IMGS': N_IMGS, 'N_TEST_IMGS': N_TEST_IMGS,'err_test':err_test,'err':err,'class_err':class_err,'class_err_test':class_err_test,'epoch_err':epoch_err})
			epoch_err_t += err[-1]
	epoch_err.append(epoch_err_t)
	print '------------ epoch err ----------'
	print epoch_err_t
sf()
