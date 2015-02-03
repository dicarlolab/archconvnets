import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
import time
import random
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import F_prod_inds
import gnumpy as gpu
gpu.board_id_to_use = 1

N = 48
N_INDS_KEEP = 20000
N_TEST_IMGS = 1024*2
filename = '/home/darren/linear_fit_imgnet_' + str(N) + '_' + str(N_INDS_KEEP) +'.mat'

z = loadmat('/home/darren/sigmas_imgnet_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat')
y = loadmat('/home/darren/patches_imgnet_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat')

sigma31 = z['sigma31']
sigma11 = z['sigma11']

sigma31_test_imgs = y['patches'][:N_TEST_IMGS]
labels = np.squeeze(y['labels'])[:N_TEST_IMGS]

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
IMG_SZ = 138 # input image size (px)
PAD = 2

n1 = N # L1 filters
n2 = N # ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 999 # number of categories

Y = np.eye(N_C)

output_sz1 = len(range(0, (IMG_SZ + 2*PAD) - s1 + 1, STRIDE1))
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

np.random.seed(6666)
inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)

FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)

sigma_inds = [0,2]
F_inds = [1,2]

EPS = 2.5e-14#2.5e-14

Y_test = np.zeros((N_C, sigma31_test_imgs.shape[0]))
Y_test[labels, range(sigma31_test_imgs.shape[0])] = 1

class_train = []
class_test = []

err_train = []
err_test = []

sigma31_g = gpu.garray(sigma31)
sigma11_g = gpu.garray(sigma11)

print 'starting'
########
t_start = time.time()
for step in range(100000):
	FL321_g = gpu.garray(FL321)
	
	grad = 2*(FL321_g.dot(sigma11_g) - sigma31_g)
	
	FL321 -= EPS * grad.as_numpy_array()
	
	if (step % 50) == 0:
		pred = np.einsum(sigma31_test_imgs, sigma_inds, FL321, F_inds, [1,0])
		err_test.append(np.mean((pred - Y_test)**2))
		class_test.append((np.argmax(pred,axis=0) == labels).sum()/np.single(N_TEST_IMGS))
		
		print err_test[-1], 1 - class_test[-1], time.time() - t_start, filename
		savemat(filename, {'err_test': err_test, 'err_train': err_train, 
			'class_test': class_test, 'class_train': class_train})
		t_start = time.time()
	
