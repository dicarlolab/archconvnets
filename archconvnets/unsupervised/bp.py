#import numpy as npd
import time
import numpy as np
from archconvnets.unsupervised.conv import conv_block
from archconvnets.unsupervised.pool_inds import max_pool_locs
from archconvnets.unsupervised.compute_L1_grad import L1_grad
from archconvnets.unsupervised.compute_L2_grad import L2_grad
from archconvnets.unsupervised.compute_L3_grad import L3_grad
from archconvnets.unsupervised.compute_FL_grad import FL_grad
from scipy.io import savemat

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

eps_F1 = 1e-2#1e-7
eps_F2 = 1e-2#1e-7
eps_F3 = 1e-3#1e-8
eps_FL = 1e-2

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 2 # batch size
N_TEST_IMGS = 100
IMG_SZ = 42 # input image size (px)

N = 4 
n1 = N # L1 filters
n2 = N # ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 10 # number of categories

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

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')

err = []
class_err = []
err_test = []
class_err_test = []
for step in range(np.int(1e7)):
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
	class_err_test.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y,axis=0)))/N_TEST_IMGS)
	
	t_test_forward_start = time.time() - t_test_forward_start

	################### compute train err
	# load imgs
	x = z['data'] - imgs_mean
	x = x.reshape((3, 32, 32, 10000))
	x = x[:,:,:,step*N_IMGS:(step+1)*N_IMGS]

	l = np.zeros((N_IMGS, N_C),dtype='int')
	l[np.arange(N_IMGS),np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)] = 1
	Y = np.double(l.T)

	imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS))
	imgs_pad[:,5:5+32,5:5+32] = x


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
	class_err.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y,axis=0)))/N_IMGS)

	output_switches1_x *= STRIDE1
	output_switches1_y *= STRIDE1
	
	t_forward_start = time.time() - t_forward_start
	
	########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_

	t_grad_start = time.time()
	grad = L1_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred, Y, imgs_pad)
	F1 -= eps_F1 * grad
	
	########### F2 deriv wrt f2_, f1_, a2_x_, a2_y_

	grad = L2_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred, Y, imgs_pad)
	F2 -= eps_F2 * grad
	
	########### F3 deriv wrt f3_, f2_, a3_x_, a3_y_

	grad = L3_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred, Y, imgs_pad)
	F3 -= eps_F3 * grad
	
	########### FL deriv wrt cat_, f3_, z1_, z2_

	t_grad_start = time.time()
	grad = FL_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred, Y, imgs_pad)
	FL -= eps_FL * grad
	
	#######################################
	
	savemat('/home/darren/cifar_F1t.mat', {'F1': F1, 'F2': F2, 'F3':F3, 'eps_F3': eps_F3, 'eps_F2': eps_F2, 'step': step, 'eps_F1': eps_F1, 'N_IMGS': N_IMGS, 'N_TEST_IMGS': N_TEST_IMGS,'err_test':err_test,'err':err,'class_err':class_err,'class_err_test':class_err_test})
	print step, err_test[-1], class_err_test[-1], err[-1], class_err[-1], time.time() - t_grad_start

