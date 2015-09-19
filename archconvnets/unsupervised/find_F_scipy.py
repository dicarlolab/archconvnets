import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
import time
import random
import scipy

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


np.random.seed(62)
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

F1_shape = F1.shape
F2_shape = F2.shape
F3_shape = F3.shape
FL_shape = FL.shape

F1_len = np.prod(F1_shape)
F2_len = np.prod(F2_shape)
F3_len = np.prod(F3_shape)
FL_len = np.prod(FL_shape)

FL321 = loadmat('FL321.mat')['FL321']*1e6

EPS = 5e8#5e9
lambd = 0

Y = np.eye(N_C)
sigma_inds = [0,2]
F_inds = [1,2]

sigma31 = sigma31.reshape((N_C, np.prod(sigma31.shape[1:])))

def obj(x):
	ind = 0
	F1 = x[:F1_len].reshape(F1_shape)
	ind += F1_len
	F2 = x[ind:ind+F2_len].reshape(F2_shape)
	ind += F2_len
	F3 = x[ind:ind+F3_len].reshape(F3_shape)
	ind += F3_len
	FL = x[ind:ind+FL_len].reshape(FL_shape)
	
	FL321_current = FL*F3*F2*F1
	diff = FL321 - FL321_current
	err = np.sum(diff**2)
	print err
	savemat('/home/darren/F1_scipy_iters.mat',{'F1':np.squeeze(F1)})
	return err

def grad(x):
	ind = 0
	F1 = x[:F1_len].reshape(F1_shape)
	ind += F1_len
	F2 = x[ind:ind+F2_len].reshape(F2_shape)
	ind += F2_len
	F3 = x[ind:ind+F3_len].reshape(F3_shape)
	ind += F3_len
	FL = x[ind:ind+FL_len].reshape(FL_shape)
	
	FL321_current = FL*F3*F2*F1
	diff = FL321 - FL321_current
	grad_F1 = -2*(F2*F3*FL* diff).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1).sum(0)
	grad_F2 = -2*(F1*F3*FL* diff).sum(0).sum(1).sum(1).sum(1).sum(-1).sum(-1).sum(-1).sum(-1).sum(-1)
	grad_F3 = -2*(F1*F2*FL* diff).sum(0).sum(0).sum(0).sum(0).sum(0).sum(1).sum(1).sum(-1).sum(-1)
	grad_FL = -2*(F1*F2*F3* diff).sum(1).sum(1).sum(1).sum(1).sum(1).sum(1).sum(1).sum(-3).sum(-3)
	
	return np.concatenate((grad_F1.ravel(), grad_F2.ravel(), grad_F3.ravel(), grad_FL.ravel()))

x0 = np.concatenate((F1.ravel(), F2.ravel(), F3.ravel(), FL.ravel()))*1e3
z = scipy.optimize.minimize(obj, x0, jac=grad, tol=1e-30, options={'maxiter':1000000})

print obj(x0), obj(z['x'])

x = z['x']
ind = 0
F1 = np.squeeze(x[:F1_len].reshape(F1_shape))

savemat('/home/darren/F1_scipy_iters.mat',{'F1':F1})
