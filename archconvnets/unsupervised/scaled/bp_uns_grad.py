#import numpy as npd
import time
import numpy as np
from archconvnets.unsupervised.conv import conv_block
from archconvnets.unsupervised.pool_inds import max_pool_locs
from archconvnets.unsupervised.pool_alt_inds import max_pool_locs_alt
from archconvnets.unsupervised.scaled.compute_L1_grad import L1_grad
from archconvnets.unsupervised.scaled.compute_L2_grad import L2_grad
from archconvnets.unsupervised.scaled.compute_L3_grad import L3_grad
from archconvnets.unsupervised.scaled.compute_FL_grad import FL_grad
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random

filename = '/home/darren/cifar_test.mat'

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

EPS = 5e-6#1e-11#e-3#-2#6#7
#EPS = 1e-3
eps_F1 = EPS
eps_F2 = EPS
eps_F3 = 0.1*EPS
eps_FL = EPS

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 1 # batch size
N_TEST_IMGS = 50
IMG_SZ = 64 # input image size (px)

N = 4
n1 = N # L1 filters
n2 = N # ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 999 # number of categories

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE))

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE))

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

if False:
	x = loadmat('/home/darren/cifar_F1t_avg_a12.mat')
	F1 = x['F1']
	F2 = x['F2']
	F3 = x['F3']
	FL = x['FL']
	class_err = x['class_err'].tolist()
	class_err_test = x['class_err_test'].tolist()
	err = x['err'].tolist()
	err_test = x['err_test'].tolist()
else:
	np.random.seed(216)
	F1 = np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1))
	F1_init = copy.deepcopy(F1)
	F2 = np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2))
	F2_init = copy.deepcopy(F2)
	F3 = np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3))
	F3_init = copy.deepcopy(F3)
	FL = np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3))
	FL_init = copy.deepcopy(FL)
	err = []
	class_err = []
	err_test = []
	class_err_test = []

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')

######### sigma 31
'''d = z['data'] - imgs_mean
avg_imgs = np.zeros((3072,10))
for cat in range(10):
    avg_imgs[:,cat] = np.mean(z['data'][:,np.asarray(z['labels']) == cat],axis=1)
sigma31 = np.zeros((10,3,IMG_SZ,IMG_SZ))
for cat in range(10):
    sigma31[cat,:,5:5+32,5:5+32] = avg_imgs[:,cat].reshape((3,32,32))
'''
#sigma31 = loadmat('/home/darren/sigma31.mat')['sigma31'] / (N_IMGS * 2000)
#sigma31 = loadmat('/home/darren/sigma31_full_256_16.mat')['sigma31'] / (N_IMGS * 256)
#sigma31 = sigma31.astype('double')
sigma31 = np.random.random((10,3,n1,5,5))

#sigma31 = np.zeros_like(sigma31)
#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.tile(sigma31[:,:,:,:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis],(1,1,1,1,1,1,1,1,4,3,3))

'''sigma31 = np.mean(sigma31,axis=-1)
sigma31 = np.mean(sigma31,axis=-1)
sigma31 = np.mean(sigma31,axis=-1)
sigma31 = np.mean(sigma31,axis=-1)
sigma31 = np.mean(sigma31,axis=-1)
sigma31 = np.mean(sigma31,axis=-1)'''
#sigma31 = np.tile(sigma31[:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis],(1,1,1,1,1,4,5,5,4,3,3))


#sigma31 = np.mean(sigma31,axis=0)
#sigma31 = np.tile(sigma31[np.newaxis],(10,1,1,1,1,1,1,1,1,1,1))

#sigma31 = np.mean(sigma31,axis=2)
#sigma31 = np.tile(sigma31[:,:,np.newaxis],(1,1,4,1,1,1,1,1,1,1,1))

#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.mean(sigma31,axis=-1)
#sigma31 = np.mean(sigma31,axis=2)
#sigma31 = np.tile(sigma31[:,:,np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis],(1,1,4,1,1,4,5,5,4,3,3))

#sigma31 = np.zeros_like(sigma31)

#F1 = zscore(F1,axis=None)
#F2 = zscore(F2,axis=None)
#F3 = zscore(F3,axis=None)
#FL = zscore(FL,axis=None)

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
	err_test.append(np.sum((pred - Y)**2)/N_TEST_IMGS)
	class_err_test.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y,axis=0)))/N_TEST_IMGS)
	
	t_test_forward_start = time.time() - t_test_forward_start

	################### compute train err
	# load imgs
	x = z['data'] - imgs_mean
	x = x.reshape((3, 32, 32, 10000))
	x = x[:,:,:,step*N_IMGS:(step+1)*N_IMGS]

	l = np.zeros((N_IMGS, N_C),dtype='int')
	img_cats = np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)
	l[np.arange(N_IMGS),np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)] = 1
	Y = np.double(l.T)

	imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS))
	imgs_pad[:,5:5+32,5:5+32] = x


	# forward pass init filters
	t_forward_start = time.time()
	
	conv_output1_init = conv_block(F1_init.transpose((1,2,3,0)), imgs_pad, stride=STRIDE1)
	max_output1, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1_init)

	conv_output2_init = conv_block(F2_init.transpose((1,2,3,0)), max_output1)
	max_output2, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2_init)

	conv_output3_init = conv_block(F3_init.transpose((1,2,3,0)), max_output2)
	max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3_init)
	
	# forward pass current filters with initial switches
	'''conv_output1 = conv_block(F1.transpose((1,2,3,0)), imgs_pad, stride=STRIDE1)
	max_output1, output_switches1_x, output_switches1_y = max_pool_locs_alt(conv_output1, conv_output1_init)

	conv_output2 = conv_block(F2.transpose((1,2,3,0)), max_output1)
	max_output2, output_switches2_x, output_switches2_y = max_pool_locs_alt(conv_output2, conv_output2_init)

	conv_output3 = conv_block(F3.transpose((1,2,3,0)), max_output2)
	max_output3, output_switches3_x, output_switches3_y = max_pool_locs_alt(conv_output3, conv_output3_init)'''
	
	conv_output1 = conv_block(F1.transpose((1,2,3,0)), imgs_pad, stride=STRIDE1)
	max_output1, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)

	conv_output2 = conv_block(F2.transpose((1,2,3,0)), max_output1)
	max_output2, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)

	conv_output3 = conv_block(F3.transpose((1,2,3,0)), max_output2)
	max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)
	
	pred = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_IMGS)))
	err.append(np.sum((pred - Y)**2)/N_IMGS)
	class_err.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y,axis=0)))/N_IMGS)

	output_switches1_x *= STRIDE1
	output_switches1_y *= STRIDE1
	
	t_forward_start = time.time() - t_forward_start
	
	########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_
	t_grad_start = time.time()
	
	# output_switches2_x.shape = (32,11,11,1)
	pool1_patches = np.zeros((n1, max_output_sz1, max_output_sz1, N_IMGS, 3,s1,s1))
	for f1 in range(n1):
		for z1 in range(max_output_sz1):
			for z2 in range(max_output_sz1):
				for img in range(N_IMGS):
					# pool1 -> conv1
					a1_x_global = output_switches1_x[f1, z1, z2, img]
					a1_y_global = output_switches1_y[f1, z1, z2, img]
					# conv1 -> px
					pool1_patches[f1, z1, z2, img] = imgs_pad[:, a1_x_global:a1_x_global+s1, a1_y_global:a1_y_global+s1, img]
	
	grad = np.zeros_like(F1)
	for a1_x_ in range(s1):
		for a1_y_ in range(s1):
			for channel_ in range(3):
				pool1_deriv = pool1_patches[:,:,:,:,channel_,a1_x_,a1_y_]
				
				for f1_ in range(n1):
					conv_output2_deriv = conv_block(F2.transpose((1,2,3,0))[f1_][np.newaxis], pool1_deriv[f1_][np.newaxis])
					max_output2, output_switches2_x, output_switches2_y = max_pool_locs_alt(conv_output2_deriv, conv_output2)
					
					conv_output3_deriv = conv_block(F3.transpose((1,2,3,0)), max_output2)
					max_output3, output_switches3_x, output_switches3_y = max_pool_locs_alt(conv_output3_deriv, conv_output3)
					
					pred_deriv = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_IMGS))).T
					
					grad[f1_, channel_, a1_x_, a1_y_] = np.dot(pred_deriv, pred)
	F1 -= eps_F1 * grad
	
	########## F2 deriv wrt f2_, a2_x_, a2_y_, f1_
	# output_switches2_x.shape = (32,11,11,1)
	pool2_patches = np.zeros((n2, max_output_sz2, max_output_sz2, N_IMGS, n1,s2,s2))
	for f2 in range(n2):
		for z1 in range(max_output_sz2):
			for z2 in range(max_output_sz2):
				for img in range(N_IMGS):
					# pool2 -> conv2
					a2_x_global = output_switches2_x[f2, z1, z2, img]
					a2_y_global = output_switches2_y[f2, z1, z2, img]
					# conv2 -> pool1
					pool2_patches[f2, z1, z2, img] = max_output1[:, a2_x_global:a2_x_global+s2, a2_y_global:a2_y_global+s2, img]
	
	grad = np.zeros_like(F2)
	for a2_x_ in range(s2):
		for a2_y_ in range(s2):
			for f1_ in range(n1):
				pool2_deriv = pool2_patches[:,:,:,:,f1_,a2_x_,a2_y_]
				
				for f2_ in range(n2):
					conv_output3_deriv = conv_block(F3.transpose((1,2,3,0))[f2_][np.newaxis], pool2_deriv[f2_][np.newaxis])
					max_output3, output_switches3_x, output_switches3_y = max_pool_locs_alt(conv_output3_deriv, conv_output3)
					
					pred_deriv = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_IMGS))).T
					
					grad[f2_, f1_, a2_x_, a2_y_] = np.dot(pred_deriv, pred)
	F2 -= eps_F2 * grad

	########## F3 deriv wrt f3_, a3_x_, a3_y_, f2_
	# output_switches3_x.shape = (n3,3,3,1)
	# conv_output2.shape = (n2,25,25,1)
	pool3_patches = np.zeros((n3, max_output_sz3, max_output_sz3, N_IMGS, n2,s3,s3))
	for f3 in range(n3):
		for z1 in range(max_output_sz3):
			for z2 in range(max_output_sz3):
				for img in range(N_IMGS):
					# pool3 -> conv3
					a3_x_global = output_switches3_x[f3, z1, z2, img]
					a3_y_global = output_switches3_y[f3, z1, z2, img]
					# conv3 -> pool2
					pool3_patches[f3, z1, z2, img] = max_output2[:, a3_x_global:a3_x_global+s3, a3_y_global:a3_y_global+s3, img]
	
	grad = np.zeros_like(F3)
	for a3_x_ in range(s3):
		for a3_y_ in range(s3):
			for f2_ in range(n2):
				pool3_deriv = pool3_patches[:,:,:,:,f2_,a3_x_,a3_y_]
				for f3_ in range(n3):
					pred_deriv = np.dot(FL[:,f3_].reshape((N_C, max_output_sz3**2)), pool3_deriv[f3_].reshape((max_output_sz3**2, N_IMGS))).T
					
					grad[f3_, f2_, a3_x_, a3_y_] = np.dot(pred_deriv, pred)
					
	F3 -= eps_F3 * grad
	
	########## FL deriv wrt cat_, f3_, z1_, z2_
	grad = np.tile((pred[:,np.newaxis,np.newaxis,np.newaxis]*max_output3[np.newaxis]).sum(0).sum(-1)[np.newaxis], (999,1,1,1))
	FL -= eps_FL * grad
	
	'''grad = L1_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred, Y, imgs_pad, sigma31, img_cats)
	#print np.max(np.abs(grad))
	F1 -= eps_F1 * grad #/ np.max(grad) #zscore(grad,axis=None)#grad / np.std(grad)
	#F1 = zscore(F1,axis=None)
	print 'l1'
	########### F2 deriv wrt f2_, f1_, a2_x_, a2_y_

	grad = L2_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred, Y, imgs_pad, sigma31, img_cats)
	#print np.max(np.abs(grad))
	F2 -= eps_F2 * grad #/ np.max(grad) #* zscore(grad,axis=None)#grad / np.std(grad)
	#F2 = zscore(F2,axis=None)
	print 'l2'
	########### F3 deriv wrt f3_, f2_, a3_x_, a3_y_

	grad = L3_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred, Y, imgs_pad, sigma31, img_cats)
	#print np.max(np.abs(grad))
	F3 -= eps_F3 * grad #/ np.max(grad) ##* zscore(grad,axis=None)#grad / np.std(grad)
	#F3 = zscore(F3,axis=None)
	print 'l3'
	########### FL deriv wrt cat_, f3_, z1_, z2_

	grad = FL_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred, Y, imgs_pad, sigma31, img_cats)
	#print np.max(np.abs(grad))
	FL -= eps_FL * grad #/ np.max(grad) ##* grad / np.std(grad)
	#FL = zscore(FL,axis=None)
	print 'fl'
	#######################################
	'''
	savemat(filename, {'F1': F1, 'F2': F2, 'F3':F3, 'FL': FL, 'eps_FL': eps_FL, 'eps_F3': eps_F3, 'eps_F2': eps_F2, 'step': step, 'eps_F1': eps_F1, 'N_IMGS': N_IMGS, 'N_TEST_IMGS': N_TEST_IMGS,'err_test':err_test,'err':err,'class_err':class_err,'class_err_test':class_err_test})
	print step, err_test[-1], class_err_test[-1], err[-1], class_err[-1], time.time() - t_grad_start, filename, t_forward_start

