import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
import time
import random
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import F_prod_inds
from scipy.stats import pearsonr
import copy
import gnumpy as gpu

#kernprof -l linear_fit_cifar_mcc.py
#python -m line_profiler linear_fit_cifar_mcc.py.lprof  > p
#@profile
#def sf():
N = 16
N_INDS_KEEP = 3000
N_TRAIN = 20000
N_TEST = 1000
TOP_N = 1

N_C = 1000 # number of categories

filename = '/home/darren/linear_fit_' + str(N) + '_' + str(N_INDS_KEEP) + '_' + str(N_TRAIN) + '_' + str(N_TEST) + '_patches_' + str(N_C) + '.mat'

z = loadmat('/home/darren/sigmas_train_test_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat')

sigma31 = z['patches'][:N_C] * 5
#sigma31 = z['sigma31']
sigma31_test_imgs = z['patches']
sigma11 = z['sigma11']
inds_keep = np.squeeze(z['inds_keep'])

labels = np.zeros(60000,dtype='int')
for batch in range(1,7):
	z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
	labels[(batch-1)*10000:batch*10000] = np.squeeze(np.asarray(z['labels']).astype(int))

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
IMG_SZ = 34 # input image size (px)
PAD = 2

n1 = N # L1 filters
n2 = N # ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

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

FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
fl321s = FL321.shape
FL321 = FL321.ravel()
random.shuffle(FL321)
FL321 = FL321.reshape(fl321s)

sigma_inds = [0,2]
F_inds = [1,2]

EPS = 2e-8

Y_test = np.zeros((10, len(labels)))
Y_test[labels, range(len(labels))] = 1

class_test = []
err_test = []

sigma31_g = gpu.garray(sigma31)
sigma11_g = gpu.garray(sigma11)

print 'starting'
########
step = 0
t_start = time.time()

pred_train = zscore(np.einsum(sigma31_test_imgs[:N_TRAIN], sigma_inds, FL321, F_inds, [0,1]), axis=1)
pred = zscore(np.einsum(sigma31_test_imgs[60000-N_TEST:], sigma_inds, FL321, F_inds, [0,1]), axis=1)

test_corrs = np.dot(pred, pred_train.T)
hit = 0
for test_img in range(N_TEST):
	hit += np.max(labels[60000-1-test_img] == labels[np.argsort(-test_corrs[test_img])[:TOP_N]])

class_test.append(1 - hit/np.single(N_TEST))
print class_test[-1], time.time() - t_start, filename

while True:
	FL321_g = gpu.garray(FL321)
	
	grad = 2*(FL321_g.dot(sigma11_g) - sigma31_g)
	
	FL321 -= EPS * grad.as_numpy_array()
	if (step % 50) == 0:
		pred_train = zscore(np.einsum(sigma31_test_imgs[:N_TRAIN], sigma_inds, FL321, F_inds, [0,1]), axis=1)
		pred = zscore(np.einsum(sigma31_test_imgs[60000-N_TEST:], sigma_inds, FL321, F_inds, [0,1]), axis=1)
		
		test_corrs = np.dot(pred, pred_train.T)
		hit = 0
		for test_img in range(N_TEST):
			hit += np.max(labels[60000-N_TEST + test_img] == labels[np.argsort(-test_corrs[test_img])[:TOP_N]])
		
		class_test.append(1 - hit/np.single(N_TEST))
		print class_test[-1], time.time() - t_start, filename
		
		savemat(filename, {'class_test': class_test})
		t_start = time.time()
	step += 1
sf()
