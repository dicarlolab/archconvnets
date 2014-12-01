#import numpy as npd
import time
import numpy as np
from archconvnets.unsupervised.conv import conv_block
from archconvnets.unsupervised.pool_inds import max_pool_locs
from archconvnets.unsupervised.compute_L1_grad import L1_grad
from scipy.io import savemat

F1_scale = 0.1
F2_scale = 0.1
F3_scale = 0.1
FL_scale = 0.1

eps_F1 = 1e-7#1e-12
POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1
N_IMGS = 2
N_TEST_IMGS = 10
IMG_SZ = 42
N = 16
n1 = N
n2 = N
n3 = N

s3 = 3
s2 = 5
s1 = 5

N_C = 10# number of categories

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE))

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE))

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

F1 = np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1))
F2 = np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2))
F3 = np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3))
FL = np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3))
#imgs = np.random.random((3, IMG_SZ, IMG_SZ, N_IMGS)) - 0.5
#Y = np.random.random((N_C, N_IMGS))

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')

err = []
err_test = []
for step in range(10):
	# load imgs
	x = z['data'] - imgs_mean
	x = x.reshape((3, 32, 32, 10000))
	x = x[:,:,:,step*N_IMGS:(step+1)*N_IMGS]

	l = np.zeros((N_IMGS, N_C),dtype='int')
	l[np.arange(N_IMGS),np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)] = 1
	Y = np.double(l.T)

	imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS))
	imgs_pad[:,5:5+32,5:5+32] = x

	Y_cat_sum = Y.sum(0)

	# forward pass
	t_forward_start = time.time()
	conv_output1 = conv_block(F1.transpose((1,2,3,0)), imgs_pad, stride=STRIDE1)
	max_output1, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)

	conv_output2 = conv_block(F2.transpose((1,2,3,0)), max_output1)
	max_output2, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)

	conv_output3 = conv_block(F3.transpose((1,2,3,0)), max_output2)
	max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)

	pred = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_IMGS)))
	err.append(np.sum((pred - Y)**2))

	pred_cat_sum = pred.sum(0) # sum over categories

	output_switches1_x *= STRIDE1
	output_switches1_y *= STRIDE1
	
	t_forward_start = time.time() - t_forward_start
	
	########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_

	t_grad_start = time.time()
	grad = L1_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred_cat_sum, Y_cat_sum, imgs_pad)
	
	########################## compute test err
	# load imgs
	x = z['data'] - imgs_mean
	x = x.reshape((3, 32, 32, 10000))
	x = x[:,:,:,10000-N_TEST_IMGS:]

	l = np.zeros((N_TEST_IMGS, N_C),dtype='int')
	l[np.arange(N_TEST_IMGS),np.asarray(z['labels'])[10000-N_TEST_IMGS:].astype(int)] = 1
	Y = np.double(l.T)

	imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_TEST_IMGS))
	imgs_pad[:,5:5+32,5:5+32] = x

	# forward pass
	t_test_forward_start = time.time()
	conv_output1 = conv_block(F1.transpose((1,2,3,0)), imgs_pad, stride=STRIDE1)
	max_output1, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)

	conv_output2 = conv_block(F2.transpose((1,2,3,0)), max_output1)
	max_output2, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)

	conv_output3 = conv_block(F3.transpose((1,2,3,0)), max_output2)
	max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)

	pred = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_TEST_IMGS)))
	err_test.append(np.sum((pred - Y)**2))
	
	t_test_forward_start = time.time() - t_test_forward_start
	#######################################
	
	F1 -= eps_F1 * grad
	savemat('/home/darren/cifar_F1.mat', {'F1': F1, 'step': step, 'eps_F1': eps_F1, 'N_IMGS': N_IMGS, 'N_TEST_IMGS': N_TEST_IMGS})
	print step, err_test[-1], err[-1], t_test_forward_start, t_forward_start, time.time() - t_grad_start

