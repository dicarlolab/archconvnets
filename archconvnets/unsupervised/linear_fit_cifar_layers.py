import numpy as np
from scipy.stats import zscore
from scipy.io import savemat, loadmat
from scipy.spatial.distance import squareform
import time
import random
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import F_prod_inds, F_layer_sum_deriv_inds_gpu, F_layer_sum_deriv_inds_gpu_return, F_layer_sum_inds, set_sigma11_buffer, set_FL321_buffer

#kernprof -l linear_fit_cifar_layers.py
#python -m line_profiler linear_fit_cifar_layers.py.lprof  > p

#@profile
#def sf():
N = 16
N_INDS_KEEP = 3000
EPS = 1e-4

save_filename = '/home/darren/linear_fit_layers_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat'

z = loadmat('/home/darren/sigmas_train_test_' + str(N) + '_' + str(N_INDS_KEEP) + '.mat')

sigma31 = z['sigma31']
sigma31est_imgs = z['patches'][50000:]
inds_keep = np.squeeze(z['inds_keep'])
labels = np.squeeze(z['labels'])#[50000:]
sigma11 = np.ascontiguousarray(np.squeeze(z['sigma11']))

F1_scale = 0.0001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

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

sigma_inds = [0,2]
F_inds = [1,2]

Y_test = np.zeros((N_C, sigma31est_imgs.shape[0]))
Y_test[labels, range(sigma31est_imgs.shape[0])] = 1

class_test = []
err_test = []

t_start = time.time()

step = 0

sigma11_len = 0.5*(N_INDS_KEEP-1)*N_INDS_KEEP + N_INDS_KEEP
sigma11_lin = np.zeros(sigma11_len, dtype='single')

sigma11_triangle = squareform(sigma11,checks=False)
sigma11_lin[:len(sigma11_triangle)] = sigma11_triangle
sigma11_lin[len(sigma11_triangle):] = sigma11[range(N_INDS_KEEP),range(N_INDS_KEEP)]

for gpu in range(1,4):
	set_sigma11_buffer(sigma11_lin, inds_keep, gpu)


while True:	
	t = time.time()
	
	FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
	for gpu in range(1,4):
		set_FL321_buffer(FL321, gpu)
	
	FL32 = F_prod_inds(np.ones_like(F1), F2, F3, FL, inds_keep)
	FL31 = F_prod_inds(F1, np.ones_like(F2), F3, FL, inds_keep)
	FL21 = F_prod_inds(F1, F2, np.ones_like(F3), FL, inds_keep)
	F321 = F_prod_inds(F1, F2, F3, np.ones_like(FL), inds_keep)
	
	F_layer_sum_deriv_inds_gpu(FL32, F1, F2, F3, FL, 1, 3)
	F_layer_sum_deriv_inds_gpu(FL31, F1, F2, F3, FL, 2, 1)
	F_layer_sum_deriv_inds_gpu(FL21, F1, F2, F3, FL, 3, 2)
	F_layer_sum_deriv_inds_gpu(F321, F1, F2, F3, FL, 4, 3)
	
	s_F1 = F_layer_sum_inds(FL32*sigma31, F1, F2, F3, FL, inds_keep, 1)
	s_F2 = F_layer_sum_inds(FL31*sigma31, F1, F2, F3, FL, inds_keep, 2)
	s_F3 = F_layer_sum_inds(FL21*sigma31, F1, F2, F3, FL, inds_keep, 3)
	s_FL = F_layer_sum_inds(F321*sigma31, F1, F2, F3, FL, inds_keep, 4)
	
	grad_F1 = F_layer_sum_deriv_inds_gpu_return(1,3) - s_F1
	grad_F2 = F_layer_sum_deriv_inds_gpu_return(2,1) - s_F2
	grad_F3 = F_layer_sum_deriv_inds_gpu_return(3,2) - s_F3
	grad_FL = F_layer_sum_deriv_inds_gpu_return(4,3) - s_FL
	
	F1 -= EPS*grad_F1
	F2 -= EPS*grad_F2
	F3 -= EPS*grad_F3
	FL -= EPS*grad_FL
	
	if (step % 15) == 0:
		FL321 = F_prod_inds(F1, F2, F3, FL, inds_keep)
		pred = np.einsum(sigma31est_imgs, sigma_inds, FL321, F_inds, [1,0])
		err_test.append(np.mean((pred - Y_test)**2))
		class_test.append(1 - (np.argmax(pred,axis=0) == labels).sum()/10000.0)
		
		print err_test[-1], step, class_test[-1], time.time() - t_start, save_filename
		savemat(save_filename, {'err_test': err_test, 'F1': F1, 'class_test': class_test})
		t_start = time.time()
	step += 1
sf()
