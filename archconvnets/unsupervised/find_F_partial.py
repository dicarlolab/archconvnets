import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
import time
import random
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import F_prod_inds, F_layer_sum_inds

N = 48
N_INDS_KEEP = 10000
N_TEST_IMGS = 1024*2
filename = '/home/darren/F1_imgnet.mat'

FL321 = loadmat('/home/darren/linear_fit_imgnet_FL321_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat')['FL321']
z = loadmat('/home/darren/patches_imgnet_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat')

sigma31_test_imgs = z['patches'][:N_TEST_IMGS]
labels = np.squeeze(z['labels'])[:N_TEST_IMGS]

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
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 999 # number of categories

Y = np.eye(N_C)

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

if False:
	z = loadmat('/home/darren/F1.mat')
	F1 = np.ascontiguousarray(z['F1'])
	F2 = np.ascontiguousarray(z['F2'])
	F3 = np.ascontiguousarray(z['F3'])
	FL = np.ascontiguousarray(z['FL'])
else:
	np.random.seed(6666)
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

EPS = 5e5

Y_test = np.zeros((N_C, sigma31_test_imgs.shape[0]))
Y_test[labels, range(sigma31_test_imgs.shape[0])] = 1

sigma_inds = [0,2]
F_inds = [1,2]

class_train = []

err_train = []
F_err = []

for step in range(10000000):
	FL321_current = F_prod_inds(F1, F2, F3, FL, inds_keep)
	diff = FL321 - FL321_current
	
	grad_F1 = 2*F_layer_sum_inds(F_prod_inds(np.ones_like(F1), F2, F3, FL, inds_keep) * diff, F1, F2, F3, FL, inds_keep, 1)
	
	grad_F2 = 2*F_layer_sum_inds(F_prod_inds(F1, np.ones_like(F2), F3, FL, inds_keep) * diff, F1, F2, F3, FL, inds_keep, 2)
	
	grad_F3 = 2*F_layer_sum_inds(F_prod_inds(F1, F2, np.ones_like(F3), FL, inds_keep) * diff, F1, F2, F3, FL, inds_keep, 3)
	
	grad_FL = 2*F_layer_sum_inds(F_prod_inds(F1, F2, F3, np.ones_like(FL), inds_keep) * diff, F1, F2, F3, FL, inds_keep, 4)
	
	F1 += EPS * grad_F1
	F2 += EPS * grad_F2
	F3 += EPS * grad_F3
	FL += EPS * grad_FL
	
	if (step % 10) == 0:
		pred = np.einsum(sigma31_test_imgs, sigma_inds, FL321_current, F_inds, [1,0])
		err_train.append(np.mean((pred - Y_test)**2))
		class_train.append(1 - (np.argmax(pred,axis=0) == labels).sum()/np.single(10000))
		
		F_err.append(np.mean(diff**2))
		
		print F_err[-1], err_train[-1], class_train[-1]
		savemat(filename,{'F1':np.squeeze(F1), 'F_err':F_err,
			'F2':np.squeeze(F2), 'F3':np.squeeze(F3), 'FL':np.squeeze(FL), 'EPS':EPS,'err_train':err_train, 'class_train':class_train})
