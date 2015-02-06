import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
import time
import random
import scipy
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import F_prod_inds, F_layer_sum_deriv_inds, F_layer_sum_inds

N = 2
N_INDS_KEEP = 50

z = loadmat('/home/darren/sigmas_train_test_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat')

sigma31 = z['sigma31']
sigma31_test_imgs = z['patches']
labels = z['labels']
sigma11 = z['sigma11']

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
IMG_SZ = 32 # input image size (px)
PAD = 2

n1 = N # L1 filters
n2 = N # ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 10 # number of categories

Y = np.eye(N_C)

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

np.random.seed(26666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500
FL = zscore(FL,axis=None)/500

np.random.seed(6666)
inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)

sigma_inds = [0,2]
F_inds = [1,2]

Y_test = np.zeros((N_C, sigma31_test_imgs.shape[0]))
Y_test[labels, range(sigma31_test_imgs.shape[0])] = 1

i_ind = 0
j_ind = 0
k_ind = 1
l_ind = -1


def f(x):
	F2[i_ind, j_ind, k_ind, l_ind] = x
	
	FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
	pred = np.einsum(sigma31_test_imgs, sigma_inds, FL321, F_inds, [1,0])
	return np.sum((pred - Y_test)**2)



def g(x):
	'''F1[i_ind, j_ind, k_ind, l_ind] = x

	FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
	
	FL32 = F_prod_inds(np.ones_like(F1), F2, F3, FL, inds_keep)
	uns = F_layer_sum_deriv_inds(FL321, FL32, sigma11, F1, F2, F3, FL, inds_keep, 1)
	s = F_layer_sum_inds(FL32*sigma31, F1, F2, F3, FL, inds_keep, 1)
	grad_F1 = 2*(uns - s)
	
	return grad_F1[i_ind, j_ind, k_ind, l_ind]'''
	
	F2[i_ind, j_ind, k_ind, l_ind] = x
	
	FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
	
	FL31 = F_prod_inds(F1, np.ones_like(F2), F3, FL, inds_keep)
	uns = F_layer_sum_deriv_inds(FL321, FL31, sigma11, F1, F2, F3, FL, inds_keep, 2)
	s = F_layer_sum_inds(FL31*sigma31, F1, F2, F3, FL, inds_keep, 2)
	grad_F2 = 2*(uns - s)
	
	return grad_F2[i_ind, j_ind, k_ind, l_ind]
	
	'''F3[i_ind, j_ind, k_ind, l_ind] = x
	
	FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
	
	FL21 = F_prod_inds(F1, F2, np.ones_like(F3), FL, inds_keep)
	uns = F_layer_sum_deriv_inds(FL321, FL21, sigma11, F1, F2, F3, FL, inds_keep, 3)
	s = F_layer_sum_inds(FL21*sigma31, F1, F2, F3, FL, inds_keep, 3)
	grad_F3 = 2*(uns - s)
	
	return grad_F3[i_ind, j_ind, k_ind, l_ind]'''
	
	'''FL[i_ind, j_ind, k_ind, l_ind] = x
	
	FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
	
	F321 = F_prod_inds(F1, F2, F3, np.ones_like(FL), inds_keep)
	uns = F_layer_sum_deriv_inds(FL321, F321, sigma11, F1, F2, F3, FL, inds_keep, 4)
	s = F_layer_sum_inds(F321*sigma31, F1, F2, F3, FL, inds_keep, 4)
	grad_FL = 2*(uns - s)
	return grad_FL[i_ind, j_ind, k_ind, l_ind]'''

eps = np.sqrt(np.finfo(np.float).eps)*1e3
x = 1e-4*F2[i_ind,j_ind-1,k_ind,l_ind]; gt = g(x); gtx = scipy.optimize.approx_fprime(np.ones(1)*x, f, eps); print gt, gtx, gtx/gt
x = 1e-5*F2[i_ind,j_ind-1,k_ind,l_ind]; gt = g(x); gtx = scipy.optimize.approx_fprime(np.ones(1)*x, f, eps); print gt, gtx, gtx/gt
x = -1e-4*F2[i_ind,j_ind-1,k_ind,l_ind]; gt = g(x); gtx = scipy.optimize.approx_fprime(np.ones(1)*x, f, eps); print gt, gtx, gtx/gt
