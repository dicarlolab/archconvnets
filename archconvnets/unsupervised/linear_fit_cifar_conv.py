import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
import time
import random
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import F_prod_inds
from scipy.stats import pearsonr

def grad_corr_x_y(x,y):
	N = len(x)
	y_no_mean = y - np.mean(y)
	x_no_mean = x - np.mean(x)

	sigma_x = np.std(x) * np.sqrt(N)
	sigma_y = np.std(y) * np.sqrt(N)

	return ((y_no_mean * sigma_x) - (np.dot(x_no_mean, y_no_mean) * x_no_mean / sigma_x)) / ((sigma_x**2) * sigma_y)

N = 4
N_INDS_KEEP = 40

filename = '/home/darren/linear_fit_' + str(N) + '_' + str(N_INDS_KEEP) + '_conv.mat'

z = loadmat('/home/darren/sigmas_train_test_' + str(N) + '_' + str(N_INDS_KEEP) + '_rand.mat')

sigma31 = z['sigma31']
sigma31_test_imgs = z['patches']
labels = z['labels']
sigma11 = z['sigma11']
inds_keep = np.squeeze(z['inds_keep'])

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

N_C = 10 # number of categories

Y = np.eye(N_C)

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

np.random.seed(660662)
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
#FL321 = np.ones_like(FL321)
#random.shuffle(FL321)
FL321 = FL321.reshape(fl321s)

sigma_inds = [0,2]
F_inds = [1,2]

EPS = 1e-12
EPS_CORR = 1e-11

Y_test = np.zeros((N_C, sigma31_test_imgs.shape[0]))
Y_test[labels, range(sigma31_test_imgs.shape[0])] = 1

class_train = []
class_test = []
convolutionarity = []
err_train = []
err_test = []

import gnumpy as gpu
sigma31_g = gpu.garray(sigma31)
sigma11_g = gpu.garray(sigma11)

print 'starting'
########
step = 0
t_start = time.time()

while True:
	FL321_g = gpu.garray(FL321)
	
	grad = 2*(FL321_g.dot(sigma11_g) - sigma31_g)
	
	FL321t = FL321[0].reshape((4*3*5*5, N_INDS_KEEP))
	grad_corr = np.zeros((4*3*5*5, N_INDS_KEEP))
	total = 0
	for i in range(N_INDS_KEEP):
		for j in range(N_INDS_KEEP):
			grad_corr[:,i] += grad_corr_x_y(FL321t[:,i], FL321t[:,j])
			grad_corr[:,j] += grad_corr_x_y(FL321t[:,j], FL321t[:,i])
			total += 1
	
	FL321 -= EPS * grad.as_numpy_array()
	FL321 += EPS_CORR*(grad_corr.ravel())[np.newaxis] / total
	
	if (step % 20) == 0:
		pred = np.einsum(sigma31_test_imgs, sigma_inds, FL321, F_inds, [1,0])
		err_test.append(np.mean((pred - Y_test)**2))
		class_test.append((np.argmax(pred,axis=0) == labels).sum())
		
		### convolutionarity f0r layer 1
		FL321t = FL321[0].reshape((4*3*5*5, N_INDS_KEEP))

		intact_sum = 0
		total = 0
		for i in range(N_INDS_KEEP):
			for j in range(i+1,N_INDS_KEEP):
				intact_sum += np.abs(pearsonr(FL321t[:,i], FL321t[:,j])[0])
				total += 1
		convolutionarity.append(intact_sum/total)
		
		print err_test[-1], 1 - class_test[-1]/10000.0, time.time() - t_start, convolutionarity[-1], filename
		savemat(filename, {'err_test': err_test, 'err_train': err_train, 
			'class_test': class_test, 'class_train': class_train, 'convolutionarity':convolutionarity})
		t_start = time.time()
	
