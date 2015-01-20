from archconvnets.unsupervised.conv import conv_block
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from archconvnets.unsupervised.cudnn_module.cudnn_module import *
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import *
from archconvnets.unsupervised.pool_inds_py import max_pool_locs

conv_block_cuda = conv_block

#kernprof -l bp_cudnn_full_sigma.py
#python -m line_profiler bp_cudnn_full_sigma.py.lprof  > p
#@profile
#def sf():

filename = '/home/darren/cifar_16_full_5batches_eps4.mat'

err_test = []
class_err_test = []

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

EPS = 1e-4#1e-5
eps_F1 = EPS
eps_F2 = EPS
eps_F3 = EPS
eps_FL = EPS
WD = 0

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 2*2500 # batch size
N_TEST_IMGS = N_IMGS #N_SIGMA_IMGS #128*2
N_SIGMA_IMGS = N_IMGS
IMG_SZ_CROP = 28 # input image size (px)
IMG_SZ = 32 # input image size (px)
img_train_offset = 2
PAD = 2

N = 16
n1 = N # L1 filters
n2 = N# ...
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

F1_init = copy.deepcopy(F1)
F2_init = copy.deepcopy(F2)
F3_init = copy.deepcopy(F3)
FL_init = copy.deepcopy(FL)


if False:
	z = loadmat(filename)
	err_test = z['err_test'].tolist()[0]
	class_err_test = z['class_err_test'].tolist()[0]
	step = np.int(z['step'])
	F1 = z['F1']
	F2 = z['F2']
	F3 = z['F3']
	FL = z['FL']
	N_IMGS = np.int(z['N_IMGS'])
	N_TEST_IMGS = np.int(z['N_TEST_IMGS'])
	print 'previous EPS:', z['eps_F1'][0][0], z['eps_F2'][0][0], z['eps_F3'][0][0], z['eps_FL'][0][0]
	print 'new eps: ', EPS


print 'loading sigma'
sigma31 = np.load('/home/darren/sigma31_16_comb.npy')

set_sigma_buffer(sigma31, 1, 0)
set_sigma_buffer(sigma31, 1, 2)
set_sigma_buffer(sigma31, 1, 3)

sigma31 = None
print 'finished'

grad_L1 = 0
grad_L2 = 0
grad_L3 = 0
grad_FL = 0

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

##################
# load test imgs into buffers
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_TEST_IMGS]

l = np.zeros((N_TEST_IMGS, N_C),dtype='int')
l[np.arange(N_TEST_IMGS),np.asarray(z['labels'])[:N_TEST_IMGS].astype(int)] = 1
Y_test = np.double(l.T)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_TEST_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP]
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

# forward pass init filters on test imgs
conv_output1 = conv_block_cuda(np.double(F1.transpose((1,2,3,0))), np.double(imgs_pad.transpose((1,2,3,0)))).transpose((3,0,1,2))
max_output1t, output_switches1_x_init, output_switches1_y_init = max_pool_locs(np.single(conv_output1))
max_output1 = np.zeros((N_TEST_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv_block_cuda(np.double(F2.transpose((1,2,3,0))), np.double(max_output1.transpose((1,2,3,0)))).transpose((3,0,1,2))
max_output2t, output_switches2_x_init, output_switches2_y_init = max_pool_locs(conv_output2, PAD=2)
max_output2 = np.zeros((N_TEST_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv_block_cuda(np.double(F3.transpose((1,2,3,0))), np.double(max_output2.transpose((1,2,3,0)))).transpose((3,0,1,2))
max_output3, output_switches3_x_init, output_switches3_y_init = max_pool_locs(conv_output3, PAD=2)

Y = np.eye(N_C)


for step in range(10000000):
	t_total = time.time()
	
	set_filter_buffers(F1,F2,F3,FL,0)
	set_filter_buffers(F1,F2,F3,FL,2)
	set_filter_buffers(F1,F2,F3,FL,3)
	
	################### launch on gpus
	# def einsum_deriv_gpu(deriv_layer_ind, sigma_ind, output_ind, gpu_ind)
	
	einsum_deriv_gpu(0,1,0,3) # pred, l1

	einsum_deriv_gpu(1,1,1,0) # deriv, l1
	einsum_deriv_gpu(2,1,3,3) # deriv, l2
	einsum_deriv_gpu(3,1,5,2) # deriv, l3
	einsum_deriv_gpu(4,1,7,2) # deriv, fl
	
	
	########################## compute test err
	t_test_forward_start = time.time()
	FLr = FL.reshape((N_C, n3*max_output_sz3**2))
	
	# forward pass current filters
	conv_output1 = conv_block_cuda(np.double(F1.transpose((1,2,3,0))), np.double(imgs_pad.transpose((1,2,3,0)))).transpose((3,0,1,2))
	max_output1t = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output1[:,np.newaxis])), output_switches1_x_init, output_switches1_y_init)
	max_output1 = np.zeros((N_TEST_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
	max_output1[:,:, PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = np.squeeze(max_output1t)

	conv_output2 = conv_block_cuda(np.double(F2.transpose((1,2,3,0))), np.double(max_output1.transpose((1,2,3,0)))).transpose((3,0,1,2))
	max_output2t = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output2[:,np.newaxis])), output_switches2_x_init, output_switches2_y_init)
	max_output2 = np.zeros((N_TEST_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
	max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = np.squeeze(max_output2t)

	conv_output3 = conv_block_cuda(np.double(F3.transpose((1,2,3,0))), np.double(max_output2.transpose((1,2,3,0)))).transpose((3,0,1,2))
	max_output3 = max_pool_locs_alt(np.ascontiguousarray(np.single(conv_output3[:,np.newaxis])), output_switches3_x_init, output_switches3_y_init)

	pred = np.dot(FLr, max_output3.reshape((N_TEST_IMGS, n3*max_output_sz3**2)).T)
	err_test.append(np.mean((pred - Y_test)**2))
	class_err_test.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y_test, axis=0)))/N_TEST_IMGS)
	
	t_test_forward_start = time.time() - t_test_forward_start
	
	####### load from gpu
	
	predc = (einsum_return(0,3) - Y).reshape((N_C, N_C, 1, 1, 1, 1))
	predc2 = predc.reshape((N_C, N_C, 1, 1, 1))
	
	############################################## F1 deriv
	derivc = einsum_return(1,0)
	
	grad_L1 = 2*(derivc*predc).sum(0).sum(0)
	
	############################################# F2 deriv
	derivc = einsum_return(3,3)
	
	grad_L2 = 2*(derivc*predc).sum(0).sum(0)
	
	############################################# F3 deriv
	derivc = einsum_return(5,2)
	
	grad_L3 = 2*(derivc*predc).sum(0).sum(0)
	
	############################################# FL deriv
	derivc = einsum_return(7,2)
	
	grad_FL = 2*(predc2*derivc).sum(1)
	
	
	##########
	# weight updates
	F1 += -eps_F1 * (WD * F1 + grad_L1)
	F2 += -eps_F2 * (WD * F2 + grad_L2)
	F3 += -eps_F3 * (WD * F3 + grad_L3)
	FL += -eps_FL * (WD * FL + grad_FL)
	
	#######################################
	
	print step, err_test[-1], class_err_test[-1],  time.time() - t_total, t_test_forward_start, filename
	print '                        F1', eps_F1*np.mean(np.abs(grad_L1))/np.mean(np.abs(F1)), 'F2', eps_F2*np.mean(np.abs(grad_L2))/np.mean(np.abs(F2)), 'F3', eps_F3*np.mean(np.abs(grad_L3))/np.mean(np.abs(F3)), 'FL', eps_FL*np.mean(np.abs(grad_FL))/np.mean(np.abs(FL))
	
	print '                        F1', np.mean(np.abs(F1)), 'F2', np.mean(np.abs(F2)), 'F3', np.mean(np.abs(F3)), 'FL', np.mean(np.abs(FL)), ' m'
	savemat(filename, {'F1': F1, 'F2': F2, 'F3':F3, 'FL': FL, 'eps_FL': eps_FL, 'eps_F3': eps_F3, 'eps_F2': eps_F2, 'step': step, 'eps_F1': eps_F1, 'N_IMGS': N_IMGS, 'N_TEST_IMGS': N_TEST_IMGS,'err_test':err_test,'class_err_test':class_err_test})

sf()
