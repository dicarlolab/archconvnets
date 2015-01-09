from archconvnets.unsupervised.conv import conv_block
import time
import numpy as np
from archconvnets.unsupervised.pool_inds_py import max_pool_locs
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from archconvnets.unsupervised.cudnn_module.cudnn_module import *

conv_block_cuda = conv_block

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 12 # batch size
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


np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F1_init = copy.deepcopy(F1)
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F2_init = copy.deepcopy(F2)
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
F3_init = copy.deepcopy(F3)
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))
FL_init = copy.deepcopy(FL)
	
F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500
FL = zscore(FL,axis=None)/500
imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']


y = loadmat('/home/darren/sigma31_dbg.mat')
sigma31 = y['sigma31']
#sigma31 = sigma31.transpose((0,2,1,3,4,5,6,7,8,9,10,11,12))

# 10, n1, 3, s1, s1, n2, s2, s2, n3, s3, s3, sz, sz

sigma31_L1 = sigma31
sigma31_L2 = sigma31
sigma31_L3 = sigma31
sigma31_LF = sigma31

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

# forward pass init filters on test imgs
conv_output1 = conv_block_cuda(np.double(F1.transpose((1,2,3,0))), np.double(imgs_pad.transpose((1,2,3,0)))).transpose((3,0,1,2))
max_output1t, output_switches1_x_init, output_switches1_y_init = max_pool_locs(np.single(conv_output1))
max_output1 = np.zeros((N_TEST_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv_block_cuda(np.double(F2.transpose((1,2,3,0))), np.double(max_output1.transpose((1,2,3,0)))).transpose((3,0,1,2))
max_output2t, output_switches2_x_init, output_switches2_y_init = max_pool_locs(conv_output2, PAD=2)
max_output2 = np.zeros((N_TEST_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv_block_cuda(np.double(F3.transpose((1,2,3,0))), np.double(max_output2.transpose((1,2,3,0)))).transpose((3,0,1,2))
max_output3, output_switches3_x_init, output_switches3_y_init = max_pool_locs(conv_output3, PAD=2)


i_ind = 1
j_ind = 0
k_ind = 0
l_ind = 1

def f(x):
	FL[i_ind, j_ind, k_ind, l_ind] = x
		
	FLr = FL.reshape((N_C, n3*max_output_sz3**2))
	
	########################## compute test err
	
	# forward pass current filters
	conv_output1 = conv_block_cuda(np.double(F1.transpose((1,2,3,0))), np.double(imgs_pad.transpose((1,2,3,0)))).transpose((3,0,1,2))
	max_output1t = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output1[:,np.newaxis])), output_switches1_x_init, output_switches1_y_init)
	max_output1 = np.zeros((N_TEST_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
	max_output1[:,:, PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = np.squeeze(max_output1t)

	conv_output2 = conv_block_cuda(np.double(F2.transpose((1,2,3,0))), np.double(max_output1.transpose((1,2,3,0)))).transpose((3,0,1,2))
	max_output2t = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output2[:,np.newaxis])), output_switches2_x_init, output_switches2_y_init)
	max_output2 = np.zeros((N_TEST_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
	max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = np.squeeze(max_output2t)

	conv_output3 = conv_block_cuda(np.double(F3.transpose((1,2,3,0))), np.double(max_output2.transpose((1,2,3,0)))).transpose((3,0,1,2))
	max_output3 = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output3[:,np.newaxis])), output_switches3_x_init, output_switches3_y_init)

	pred = np.dot(FLr, max_output3.reshape((N_TEST_IMGS, n3*max_output_sz3**2)).T)
	err = np.sum((pred - Y_test)**2)# d(pred)*(pred - Y_test)
	return np.sum(err)
	#return np.sum(pred)#err
	#return np.sum(Y_test*pred)#err
	#return np.sum(pred)
	

def g(x):
	FL[i_ind, j_ind, k_ind, l_ind] = x
	
	Y = np.eye(10)
	
	FLt = FL.reshape((N_C, 1, 1, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))
	
	F21 = F1[:,:,:,:,np.newaxis,np.newaxis,np.newaxis] * F2.transpose((1,0,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
	F21 = F21.reshape((1, n1, 3, s1, s1, n2, s2, s2, 1, 1, 1, 1, 1))
	F321 = F21 * F3.transpose((1,0,2,3)).reshape((1, 1, 1, 1, 1, n2, 1, 1, n3, s3, s3, 1, 1))
	FL321 = F321 * FL.reshape((N_C, 1, 1, 1, 1, 1, 1, 1, n3, 1, 1, max_output_sz3, max_output_sz3))
	
	sigma_inds = [0,2,3,4,5,6,7,8,9,10,11,12,13]
	F_inds = [1,2,3,4,5,6,7,8,9,10,11,12,13]
	
	############################################## F1 deriv wrt f1_, a1_x_, a1_y_, channel_
	'''F32 = F2[np.newaxis,:,:,:,:,np.newaxis,np.newaxis] * F3[:,:,np.newaxis,np.newaxis,np.newaxis]
	# F32: n3, n2, n1, s2,s2, s3,s3
	F32 = F32.transpose((2,1,3,4,0,5,6))
	# F32: n1, n2, s2,s2, n3, s3,s3
	F32 = F32[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:,:,:,:,:,:,np.newaxis,np.newaxis]
	FL32 = FLt * F32
	
	sigma31_F1 = sigma31_L1 * F1.reshape((1, n1, 3, s1, s1,  1, 1, 1, 1, 1, 1, 1, 1))
	
	derivc = np.einsum(sigma31_L1, sigma_inds, FL32, F_inds, range(6))
	#grad_L1_s = derivc.sum(0).sum(0)
	predc = np.einsum(sigma31_F1, sigma_inds, FL32, F_inds, [0,1])
	grad_L1_s = np.tensordot(predc, derivc, ([0,1],[0,1]))
	grad_L1_s -= np.einsum(derivc,[0,0,2,3,4,5], [2,3,4,5])'''
	
	
	'''############################################# F2 deriv wrt f2_, f1_, a2_x_, a2_y_
	F31 = np.tensordot(F1, F3, 0).transpose((0,1,2,3,5,4,6,7))
	F31 = F31.reshape((1,n1, 3, s1, s1, n2, 1,1, n3, s3, s3, 1, 1))
	FL31 = FLt * F31
	
	sigma31_F2 = sigma31_L2 * F2.transpose((1,0,2,3)).reshape((1, n1, 1, 1, 1, n2, s2, s2, 1, 1, 1, 1, 1))
	
	derivc = np.einsum(sigma31_L2, sigma_inds, FL31, F_inds, [0,1,6,2,7,8])
	predc = np.einsum(sigma31_F2, sigma_inds, FL31, F_inds, [0,1])
	grad_L2_s = np.tensordot(predc, derivc, ([0,1],[0,1]))
	grad_L2_s -=  np.einsum(derivc,[0,0,2,3,4,5], [2,3,4,5])'''
	
	
	############################################## F3 deriv wrt f3_, f2_, a3_x_, a3_y_
	#FL21 = FLt * F21
	
	#derivc = np.einsum(sigma31_L3, sigma_inds, FL21, F_inds, [0,1,9,6,10,11])
	#predc = np.einsum(sigma31_L3, sigma_inds, FL321, F_inds, [0,1])
	#grad_L3_s = np.tensordot(predc, derivc, ([0,1],[0,1]))
	#grad_L3_s -=  np.einsum(derivc,[0,0,2,3,4,5], [2,3,4,5])
	#grad_L3_s = derivc.sum(0).sum(0)
	
	####################################### FL deriv wrt cat_, f3_, z1_, z2_
	derivc = np.einsum(sigma31_LF, sigma_inds, F321, sigma_inds, [0,9,12,13])[np.newaxis]
	predc = (np.einsum(sigma31_LF, sigma_inds, FL321, F_inds, [0,1]) - Y).T.reshape((N_C, N_C, 1, 1, 1))
	grad_FL_s = 2*(predc*derivc).sum(1)
	
	return grad_FL_s[i_ind, j_ind, k_ind, l_ind]

eps = np.sqrt(np.finfo(np.float).eps)#*1e1
#print eps
#x = FL_init[i_ind,j_ind,k_ind,l_ind]; gt = g(x); gtx = scipy.optimize.approx_fprime(np.ones(1)*x, f, eps); print gt, gtx, gtx/gt
#x = 10*FL_init[i_ind,j_ind,k_ind,l_ind]; gt = g(x); gtx = scipy.optimize.approx_fprime(np.ones(1)*x, f, eps); print gt, gtx, gtx/gt
x = 1e-4*FL_init[i_ind,j_ind,k_ind,l_ind]; gt = g(x); gtx = scipy.optimize.approx_fprime(np.ones(1)*x, f, eps); print gt, gtx, gtx/gt
x = 1e-5*FL_init[i_ind,j_ind,k_ind,l_ind]; gt = g(x); gtx = scipy.optimize.approx_fprime(np.ones(1)*x, f, eps); print gt, gtx, gtx/gt
x = -1e-4*FL_init[i_ind,j_ind,k_ind,l_ind]; gt = g(x); gtx = scipy.optimize.approx_fprime(np.ones(1)*x, f, eps); print gt, gtx, gtx/gt

#print scipy.optimize.check_grad(f,g,F1_init[i_ind,j_ind,k_ind,l_ind]*np.ones(1)), g(F1_init[i_ind,j_ind,k_ind,l_ind])
#print scipy.optimize.check_grad(f,g,-F1_init[i_ind,j_ind,k_ind,l_ind]*np.ones(1)), g(-F1_init[i_ind,j_ind,k_ind,l_ind])
#print scipy.optimize.check_grad(f,g,0.1*F1_init[i_ind,j_ind,k_ind,l_ind]*np.ones(1)), g(0.1*F1_init[i_ind,j_ind,k_ind,l_ind])
#print scipy.optimize.check_grad(f,g,10*F1_init[i_ind,j_ind,k_ind,l_ind]*np.ones(1)), g(10*F1_init[i_ind,j_ind,k_ind,l_ind])
'''print scipy.optimize.check_grad(f,g,1000*F1_init[i_ind,j_ind,k_ind,l_ind]*np.ones(1))
print scipy.optimize.check_grad(f,g,1e4*F1_init[i_ind,j_ind,k_ind,l_ind]*np.ones(1))'''



