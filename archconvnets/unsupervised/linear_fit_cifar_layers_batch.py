import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
import time
import random
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import F_prod_inds, F_layer_sum_deriv_inds, F_layer_sum_inds

#kernprof -l linear_fit_cifar_layers_batch.py
#python -m line_profiler linear_fit_cifar_layers_batch.py.lprof  > p

#@profile
#def sf():
N = 2
N_INDS_KEEP = 10000
N_INDS_UPDATE = 5000

WD = 0#5e-1

save_filename = '/home/darren/linear_fit_layers2_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat'


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

sigma_inds = [0,2]
F_inds = [1,2]

EPS = 1e-1#2.5e-4#2.5e-5#2.5e-13#2.5e-14

class_train = []
class_test = []

err_train = []
err_test = []

#inds_keep_t = inds_keep
#sigma11_t = sigma11
#sigma31_t = sigma31

t_start = time.time()
step = 0
while True:
	s_batch = step % 95
	z = loadmat('/export/imgnet_storage_full/sigma31_2_100batches/sigmas_' + str(N) + '_' + str(N_INDS_KEEP) + '_' + str(s_batch) + '.mat')

	sigma31 = z['sigma31']
	sigma31_test_imgs = z['patches']
	labels = z['labels']
	sigma11 = z['sigma11']
	
	np.random.seed(6666 + s_batch)
	inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)
	
	Y_test = np.zeros((N_C, sigma31_test_imgs.shape[0]))
	Y_test[labels, range(sigma31_test_imgs.shape[0])] = 1
	
	#F1 = zscore(F1,axis=None)/500
	#F2 = zscore(F2,axis=None)/500
	#F3 = zscore(F3,axis=None)/500
	#FL = zscore(FL,axis=None)/500
	np.random.seed(6666 + step + s_batch)
	inds_inds = np.random.randint(N_INDS_KEEP, size=N_INDS_UPDATE)
	
	inds_keep_t = inds_keep[inds_inds]
	sigma11_t = np.ascontiguousarray(sigma11[inds_inds][:, inds_inds])
	sigma31_t = np.ascontiguousarray(sigma31[:, inds_inds])
	
	FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep_t)
	
	FL32 = F_prod_inds(np.ones_like(F1), F2, F3, FL, inds_keep_t)
	uns = F_layer_sum_deriv_inds(FL321, FL32, sigma11_t, F1, F2, F3, FL, inds_keep_t, 1)
	s = F_layer_sum_inds(FL32*sigma31_t, F1, F2, F3, FL, inds_keep_t, 1)
	grad_F1 = uns - s
	
	FL31 = F_prod_inds(F1, np.ones_like(F2), F3, FL, inds_keep_t)
	uns = F_layer_sum_deriv_inds(FL321, FL31, sigma11_t, F1, F2, F3, FL, inds_keep_t, 2)
	s = F_layer_sum_inds(FL31*sigma31_t, F1, F2, F3, FL, inds_keep_t, 2)
	grad_F2 = uns - s
	
	FL21 = F_prod_inds(F1, F2, np.ones_like(F3), FL, inds_keep_t)
	uns = F_layer_sum_deriv_inds(FL321, FL21, sigma11_t, F1, F2, F3, FL, inds_keep_t, 3)
	s = F_layer_sum_inds(FL21*sigma31_t, F1, F2, F3, FL, inds_keep_t, 3)
	grad_F3 = uns - s
	
	F321 = F_prod_inds(F1, F2, F3, np.ones_like(FL), inds_keep_t)
	uns = F_layer_sum_deriv_inds(FL321, F321, sigma11_t, F1, F2, F3, FL, inds_keep_t, 4)
	s = F_layer_sum_inds(F321*sigma31_t, F1, F2, F3, FL, inds_keep_t, 4)
	grad_FL = uns - s
	
	F1 -= EPS*(grad_F1 + WD*F1)
	F2 -= EPS*(grad_F2 + WD*F2)
	F3 -= EPS*(grad_F3 + WD*F3)
	FL -= EPS*(grad_FL + WD*FL)
	
	if (step % 2) == 0:
		z = loadmat('/export/imgnet_storage_full/sigma31_2_100batches/sigmas_' + str(N) + '_' + str(N_INDS_KEEP) + '_' + str(0) + '.mat')

		sigma31 = z['sigma31']
		sigma31_test_imgs = z['patches']
		labels = z['labels']
		sigma11 = z['sigma11']
		
		np.random.seed(6666 + 0)
		inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)
		
		Y_test = np.zeros((N_C, sigma31_test_imgs.shape[0]))
		Y_test[labels, range(sigma31_test_imgs.shape[0])] = 1
	
		FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
		pred = np.einsum(sigma31_test_imgs, sigma_inds, FL321, F_inds, [1,0])
		err_test.append(np.mean((pred - Y_test)**2))
		class_test.append(1 - (np.argmax(pred,axis=0) == labels).sum()/1000.0)
		
		print err_test[-1], class_test[-1], time.time() - t_start, save_filename
		savemat(save_filename, {'err_test': err_test, 'F1': F1, 'class_test': class_test})
		t_start = time.time()
	step += 1
sf()
