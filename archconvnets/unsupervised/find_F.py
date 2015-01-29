import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
import time
import random

sigma31 = np.load('/home/darren/comb.npy')

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
IMG_SZ = 32 # input image size (px)
PAD = 2

N = 4
n1 = N # L1 filters
n2 = N# ...
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

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2))).transpose((1,0,2,3))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3))).transpose((1,0,2,3))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500
FL = zscore(FL,axis=None)/500

F1 = F1.reshape((  1, n1, 3, s1, s1,  1,  1,  1,  1,  1,  1,  1,  1))
F2 = F2.reshape((  1, n1, 1,  1,  1, n2, s2, s2,  1,  1,  1,  1,  1))
F3 = F3.reshape((  1,  1, 1,  1,  1, n2,  1,  1, n3, s3, s3,  1,  1))
FL = FL.reshape((N_C,  1, 1,  1,  1,  1,  1,  1, n3,  1,  1,  max_output_sz3, max_output_sz3))

FL321 = FL*F3*F2*F1


np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2))).transpose((1,0,2,3))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3))).transpose((1,0,2,3))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500
FL = zscore(FL,axis=None)/500

F1 = F1.reshape((  1, n1, 3, s1, s1,  1,  1,  1,  1,  1,  1,  1,  1))
F2 = F2.reshape((  1, n1, 1,  1,  1, n2, s2, s2,  1,  1,  1,  1,  1))
F3 = F3.reshape((  1,  1, 1,  1,  1, n2,  1,  1, n3, s3, s3,  1,  1))
FL = FL.reshape((N_C,  1, 1,  1,  1,  1,  1,  1, n3,  1,  1,  max_output_sz3, max_output_sz3))

FL321 = loadmat('FL321.mat')['FL321']

EPS = 7.5e9#5e9
lambd = 0

Y = np.eye(N_C)
sigma_inds = [0,2]
F_inds = [1,2]

class_train = []

err_train = []
F_err = []

sigma31 = sigma31.reshape((N_C, np.prod(sigma31.shape[1:])))

for step in range(10000000):
	FL321_current = FL*F3*F2*F1
	diff = FL321 - FL321_current
	grad_F1 = 2*(F2*F3*FL* diff).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(0)
	grad_F2 = 2*(F1*F3*FL* diff).sum(0).sum(1).sum(1).sum(1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1)
	grad_F3 = 2*(F1*F2*FL* diff).sum(0).sum(0).sum(0).sum(0).sum(0).sum(1).sum(1).sum(-1).sum(-1)
	grad_FL = 2*(F1*F2*F3* diff).sum(1).sum(1).sum(1).sum(1).sum(1).sum(1).sum(1).sum(-3).sum(-3)
	
	F1 += EPS * grad_F1.reshape(F1.shape) + lambd*F1
	F2 += EPS * grad_F2.reshape(F2.shape) + lambd*F2
	F3 += EPS * grad_F3.reshape(F3.shape) + lambd*F3
	FL += EPS * grad_FL.reshape(FL.shape) + lambd*FL
	
	pred = np.einsum(sigma31, sigma_inds, FL321_current.reshape((N_C, np.prod(FL321_current.shape[1:]))), F_inds, [1,0])
	err_train.append(np.mean((pred - Y)**2))
	class_train.append((np.argmax(pred,axis=0) == range(10)).sum())
	
	F_err.append(np.mean(diff**2))
	
	print F_err[-1], err_train[-1], class_train[-1]
	savemat('/home/darren/F1.mat',{'F1':np.squeeze(F1), 'F_err':F_err,
		'F2':np.squeeze(F2), 'F3':np.squeeze(F3), 'FL':np.squeeze(FL), 'EPS':EPS,'err_train':err_train})
