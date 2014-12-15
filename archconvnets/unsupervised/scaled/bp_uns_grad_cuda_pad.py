import theano
import theano.tensor as T
import time
import numpy as np
import numexpr as ne
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from archconvnets.unsupervised.pool_inds import max_pool_locs
from archconvnets.unsupervised.pool_alt_inds_opt import max_pool_locs_alt
from archconvnets.unsupervised.pool_inds_patch import max_pool_locs_patches
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
#kernprof -l bp_uns_grad_cuda.py
#python -m line_profiler bp_uns_grad_cuda.py.lprof  > p

#@profile
#def sf():
input = gpu_contiguous(T.tensor4('input'))
filters = gpu_contiguous(T.tensor4('filters'))

conv_op = FilterActs()
conv_block_cuda = theano.function([filters, input], conv_op(input, filters))

filename = '/home/darren/cifar_test_small2_32filters_no_sup.mat'

S_SCALE = 1e-1#1e-2
N_SIGMA_IMGS = 100
WD = 0#1e-5#1e-2 #5e-4
MOMENTUM = 0#0.9

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.001
F3_scale = 0.01
FL_scale = 0.02

EPS = 1e-2#1e-4#1e-3#1e-11#e-3#-2#6#7
#EPS = 1e-3
eps_F1 = EPS
eps_F2 = EPS
eps_F3 = EPS
eps_FL = EPS

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 200 # batch size
N_TEST_IMGS = 128*2
IMG_SZ = 42 # input image size (px)
PAD = 2

N = 32
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

if False:
	x = loadmat('/home/darren/cifar_test_small.mat')
	F1 = x['F1']
	F2 = x['F2']
	F3 = x['F3']
	FL = x['FL']
	class_err = x['class_err'].tolist()
	class_err_test = x['class_err_test'].tolist()
	err = x['err'].tolist()
	err_test = x['err_test'].tolist()
	epoch_err = x['epoch_err'].tolist()
	
	np.random.seed(623)
	F1_init = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
	F2_init = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
	F3_init = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
	FL_init = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))
else:
	np.random.seed(188521)
	F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
	F1_init = copy.deepcopy(F1)
	F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
	F2_init = copy.deepcopy(F2)
	F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
	F3_init = copy.deepcopy(F3)
	FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))
	FL_init = copy.deepcopy(FL)
	err = []
	class_err = []
	err_test = []
	class_err_test = []
	epoch_err = []

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

v_i_L1 = 0
v_i_L2 = 0
v_i_L3 = 0
v_i_FL = 0


####################
# sigma31
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')
t_sigma = time.time()
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_SIGMA_IMGS]

l = np.zeros((N_SIGMA_IMGS, N_C),dtype='int')
img_cats = np.asarray(z['labels'])[:N_SIGMA_IMGS].astype(int)
l[np.arange(N_SIGMA_IMGS),np.asarray(z['labels'])[:N_SIGMA_IMGS].astype(int)] = 1
Y = np.double(l.T)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_SIGMA_IMGS),dtype='single')
imgs_pad[:,5:5+32,5:5+32] = x

# forward pass
'''conv_output1 = np.asarray(conv_block_cuda(F1.transpose((1,2,3,0)), imgs_pad))
max_output1, output_switches1_x, output_switches1_y, pool1_patches = max_pool_locs_patches(conv_output1, imgs_pad, s1)
pool1_patches = pool1_patches.mean(1).mean(1).transpose((1,2,0,3,4)) # mean across spatial dims. new dims: [imgs x 3 x n1 x s1 x s1]

sigma31 = np.zeros((N_C, 3, n1, s1, s1))
sigma31_count = np.zeros_like(sigma31)
for img in range(N_SIGMA_IMGS):
	sigma31[img_cats[img]] += pool1_patches[img]
	sigma31_count[img_cats[img]] += 1
sigma31 /= sigma31_count'''
print 'time to compute sigma31', time.time() - t_sigma, N_SIGMA_IMGS

grad_L1_s = 0
grad_L2_s = 0
grad_L3_s = 0
grad_FL_s = 0

grad_L1_uns = 0
grad_L2_uns = 0
grad_L3_uns = 0
grad_FL_uns = 0

for iter in range(np.int(1e7)):
	epoch_err_t = 0
	for batch in range(1,6):
		for step in range(np.int((10000)/N_IMGS)):
			
			'''################################################## sigma31
			z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')
			t_sigma = time.time()
			x = z['data'] - imgs_mean
			x = x.reshape((3, 32, 32, 10000))
			x = x[:,:,:,:N_SIGMA_IMGS]

			l = np.zeros((N_SIGMA_IMGS, N_C),dtype='int')
			img_cats = np.asarray(z['labels'])[:N_SIGMA_IMGS].astype(int)
			l[np.arange(N_SIGMA_IMGS),np.asarray(z['labels'])[:N_SIGMA_IMGS].astype(int)] = 1
			Y = np.double(l.T)

			imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_SIGMA_IMGS),dtype='single')
			imgs_pad[:,5:5+32,5:5+32] = x

			# forward pass
			conv_output1 = np.asarray(conv_block_cuda(F1.transpose((1,2,3,0)), imgs_pad))
			max_output1, output_switches1_x, output_switches1_y, pool1_patches = max_pool_locs_patches(conv_output1, imgs_pad, s1)
			pool1_patches = pool1_patches.mean(1).mean(1).transpose((1,2,0,3,4)) # mean across spatial dims. new dims: [imgs x 3 x n1 x s1 x s1]

			sigma31 = np.zeros((N_C, 3, n1, s1, s1))
			sigma31_count = np.zeros_like(sigma31)
			for img in range(N_SIGMA_IMGS):
				sigma31[img_cats[img]] += pool1_patches[img]
				sigma31_count[img_cats[img]] += 1
			sigma31 /= sigma31_count
			print 'time to compute sigma31', time.time() - t_sigma, N_SIGMA_IMGS'''


			t_total = time.time()
			########################## compute test err
			# load imgs
			z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_6')
			x = z['data'] - imgs_mean
			x = x.reshape((3, 32, 32, 10000))
			x = x[:,:,:,10000-N_TEST_IMGS:]

			l = np.zeros((N_TEST_IMGS, N_C),dtype='int')
			l[np.arange(N_TEST_IMGS),np.asarray(z['labels'])[10000-N_TEST_IMGS:].astype(int)] = 1
			Y = np.double(l.T)

			imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_TEST_IMGS),dtype='single')
			imgs_pad[:,5:5+32,5:5+32] = x

			# forward pass
			t_test_forward_start = time.time()
			conv_output1 = np.asarray(conv_block_cuda(F1.transpose((1,2,3,0)), imgs_pad))
			max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)
			max_output1 = np.zeros((n1, max_output_sz1, max_output_sz1, N_TEST_IMGS),dtype='single')
			max_output1[:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

			conv_output2 = np.asarray(conv_block_cuda(F2.transpose((1,2,3,0)), max_output1))
			max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)
			max_output2 = np.zeros((n2, max_output_sz2, max_output_sz2, N_TEST_IMGS),dtype='single')
			max_output2[:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

			conv_output3 = np.asarray(conv_block_cuda(F3.transpose((1,2,3,0)), max_output2))
			max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)

			pred = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_TEST_IMGS)))
			err_test.append(np.sum((pred - Y)**2)/N_TEST_IMGS)
			class_err_test.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y,axis=0)))/N_TEST_IMGS)
			
			t_test_forward_start = time.time() - t_test_forward_start

			################### compute train err
			# load imgs
			z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
			x = z['data'] - imgs_mean
			x = x.reshape((3, 32, 32, 10000))
			x = x[:,:,:,step*N_IMGS:(step+1)*N_IMGS]

			l = np.zeros((N_IMGS, N_C),dtype='int')
			img_cats = np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)
			l[np.arange(N_IMGS),np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)] = 1
			Y = np.double(l.T)

			imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
			offset_x = 5#np.random.randint(IMG_SZ - 32 - 1)
			offset_y = 5#np.random.randint(IMG_SZ - 32 - 1)
			imgs_pad[:,offset_x:offset_x+32,offset_y:offset_y+32] = x

			t_forward_start = time.time()
			
			# forward pass current filters
			conv_output1 = np.asarray(conv_block_cuda(F1.transpose((1,2,3,0)), imgs_pad))
			max_output1t, output_switches1_x, output_switches1_y, pool1_patches = max_pool_locs_patches(conv_output1, imgs_pad, s1)
			max_output1 = np.zeros((n1, max_output_sz1, max_output_sz1, N_IMGS),dtype='single')
			max_output1[:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

			conv_output2 = np.asarray(conv_block_cuda(F2.transpose((1,2,3,0)), max_output1))
			max_output2t, output_switches2_x, output_switches2_y, pool2_patches = max_pool_locs_patches(conv_output2, max_output1, s2)
			max_output2 = np.zeros((n2, max_output_sz2, max_output_sz2, N_IMGS),dtype='single')
			max_output2[:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

			conv_output3 = np.asarray(conv_block_cuda(F3.transpose((1,2,3,0)), max_output2))
			max_output3, output_switches3_x, output_switches3_y, pool3_patches = max_pool_locs_patches(conv_output3, max_output2, s3)
			
			pred = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_IMGS)))
			err.append(np.sum((pred - Y)**2)/N_IMGS)
			class_err.append(1-np.float(np.sum(np.argmax(pred,axis=0) == np.argmax(Y,axis=0)))/N_IMGS)
			
			#pred[img_cats, range(N_IMGS)] -= 1 ############ backprop
			pred_ravel = pred.ravel()
			
			t_forward_start = time.time() - t_forward_start
			t_grad_start = time.time()
			
			########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_
			grad = np.zeros_like(F1)
			for a1_x_ in range(s1):
				for a1_y_ in range(s1):
					for channel_ in range(3):
						pool1_derivt = pool1_patches[:,:,:,:,channel_,a1_x_,a1_y_]
						pool1_deriv = np.zeros((n1, max_output_sz1, max_output_sz1, N_IMGS),dtype='single')
						pool1_deriv[:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = pool1_derivt
						for f1_ in range(n1):
							conv_output2_deriv = np.asarray(conv_block_cuda(F2.transpose((1,2,3,0))[f1_][np.newaxis], pool1_deriv[f1_][np.newaxis]))
							max_output2t = max_pool_locs_alt(conv_output2_deriv, output_switches2_x, output_switches2_y)
							max_output2[:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t
							
							conv_output3_deriv = np.asarray(conv_block_cuda(F3.transpose((1,2,3,0)), max_output2))
							max_output3 = max_pool_locs_alt(conv_output3_deriv, output_switches3_x, output_switches3_y)
							
							pred_deriv = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_IMGS))).ravel()
							
							grad[f1_, channel_, a1_x_, a1_y_] = np.dot(pred_deriv, pred_ravel)
			grad_L1_uns = grad / N_IMGS
			
			
			########## F2 deriv wrt f2_, a2_x_, a2_y_, f1_
			grad = np.zeros_like(F2)
			for a2_x_ in range(s2):
				for a2_y_ in range(s2):
					for f1_ in range(n1):
						pool2_derivt = pool2_patches[:,:,:,:,f1_,a2_x_,a2_y_]
						pool2_deriv = np.zeros((n2, max_output_sz2, max_output_sz2, N_IMGS),dtype='single')
						pool2_deriv[:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = pool2_derivt
						for f2_ in range(n2):
							conv_output3_deriv = np.asarray(conv_block_cuda(F3.transpose((1,2,3,0))[f2_][np.newaxis], pool2_deriv[f2_][np.newaxis]))
							max_output3 = max_pool_locs_alt(conv_output3_deriv, output_switches3_x, output_switches3_y)
							
							pred_deriv = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_IMGS))).ravel()
							
							grad[f2_, f1_, a2_x_, a2_y_] = np.dot(pred_deriv, pred_ravel)
			grad_L2_uns = grad / N_IMGS
			

			########## F3 deriv wrt f3_, a3_x_, a3_y_, f2_
			grad = np.zeros_like(F3)
			for a3_x_ in range(s3):
				for a3_y_ in range(s3):
					for f2_ in range(n2):
						pool3_deriv = pool3_patches[:,:,:,:,f2_,a3_x_,a3_y_]
						for f3_ in range(n3):
							pred_deriv = np.dot(FL[:,f3_].reshape((N_C, max_output_sz3**2)), pool3_deriv[f3_].reshape((max_output_sz3**2, N_IMGS))).ravel()
							
							grad[f3_, f2_, a3_x_, a3_y_] = np.dot(pred_deriv, pred_ravel)
							
			grad_L3_uns = grad / N_IMGS
			
			########## FL deriv wrt cat_, f3_, z1_, z2_
			grad_FL_uns = np.tile((pred[:,np.newaxis,np.newaxis,np.newaxis]*max_output3[np.newaxis]).sum(0).sum(-1)[np.newaxis], (N_C,1,1,1)) / N_IMGS
			
			t_grad_start = time.time() - t_grad_start
			
			t_grad_s_start = time.time()
			####################################################################### supervised:
			
			########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_
			'''sigma31_FL = (sigma31[:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis] * FL[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]).transpose((0,2,1,3,4,5,6,7))
			
			# sigma31_FL: N_C, n1, 3, s1, s1, n3, z1, z2
			F32 = F3[:,:,:,:,np.newaxis,np.newaxis,np.newaxis] * F2[np.newaxis,:,np.newaxis,np.newaxis]
			# F32: n3, n2, s3, s3, n1, s2, s2
			F32 = F32.transpose((4,0,1,2,3,5,6))
			# F32: n1, n3, n2, s3, s3, s2, s2
			F32t = F32[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
			sigma31_FLt = sigma31_FL[:,:,:,:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
			#grad_L1_s = -(ne.evaluate('sigma31_FLt * F32t')).reshape((N_C,n1,3,s1,s1,n3*(max_output_sz3**2)*n2*(s3**2)*(s2**2))).sum(-1).mean(0)
			grad_L1_s = -np.einsum(sigma31_FLt, range(13), F32t, range(13), [1,2,3,4]) / N_C
			
			########### F2 deriv wrt f2_, f1_, a2_x_, a2_y_
			F31 = F3[:,:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*F1[np.newaxis,np.newaxis,np.newaxis,np.newaxis]
			# F31: n3, n2, s3, s3, n1, 3, s1, s1
			F31 = F31.transpose((4,5,6,7,0,1,2,3))
			# F31: n1, 3, s1, s1, n3, n2, s3, s3
			F31t = F31[np.newaxis, :,:,:,:,:,np.newaxis,np.newaxis]
			sigma31_FLt = sigma31_FL[:,:,:,:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis]
			#grad_L2_s = -(ne.evaluate('sigma31_FLt * F31t')).reshape((N_C,n1,3*(s1**2)*n3*(max_output_sz3**2),n2,s3**2)).sum(2).sum(-1).mean(0).T[:,:,np.newaxis,np.newaxis]
			grad_L2_s = -np.einsum(sigma31_FLt, range(11), F31t, range(11), [1,8]).T[:,:,np.newaxis,np.newaxis] / N_C
			
			########### F3 deriv wrt f3_, f2_, a3_x_, a3_y_
			F21 = F2[:,:,:,:,np.newaxis,np.newaxis,np.newaxis] * F1[np.newaxis,:,np.newaxis,np.newaxis]
			# F21: n2, n1, s2, s2, 3, s1, s1
			F21 = F21.transpose((1,4,5,6,0,2,3))
			# F21: n1, 3, s1, s1, n2, s2, s2
			F21t = F21[np.newaxis,:,:,:,:,np.newaxis,np.newaxis,np.newaxis]
			sigma31_FLt = sigma31_FL[:,:,:,:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis]
			#grad_L3_s = -(ne.evaluate('sigma31_FLt * F21t')).reshape((N_C,n1*3*(s1**2),n3,(max_output_sz3**2),n2,(s2**2))).sum(1).sum(2).sum(-1).mean(0)[:,:,np.newaxis,np.newaxis]
			grad_L3_s = -np.einsum(sigma31_FLt, range(11), F21t, range(11), [5,8])[:,:,np.newaxis,np.newaxis] / N_C
			
			########### FL deriv wrt cat_, f3_, z1_, z2_
			F3t = F3.transpose((1,0,2,3))
			# F3t: n2, n3, s3, s3
			F3t = F3t[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
			F21t = F21[:,:,:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis]
			F321 = F3t*F21t
			# F321: n1, 3, s1, s1, n2, s2, s2, n3, s3, s3
			sigma31t = sigma31.transpose((0,2,1,3,4))[:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
			F321t = F321[np.newaxis]
			#grad_FL_s = -(ne.evaluate('sigma31t * F321t')).reshape((N_C,n1*3*(s1**2)*n2*(s2**2),n3,(s3**2))).sum(1).sum(-1)[:,:,np.newaxis,np.newaxis]
			grad_FL_s = -np.einsum(sigma31t, range(11), F321t, range(11), [0,8])[:,:,np.newaxis,np.newaxis]'''
			
			t_grad_s_start = time.time() - t_grad_s_start
			
			##########
			# weight decay
			v_i1_L1 = -eps_F1 * (WD * F1 + grad_L1_uns + grad_L1_s * S_SCALE) + MOMENTUM * v_i_L1
			v_i1_L2 = -eps_F2 * (WD * F2 + grad_L2_uns + grad_L2_s * S_SCALE) + MOMENTUM * v_i_L2
			v_i1_L3 = -eps_F3 * (WD * F3 + grad_L3_uns + grad_L3_s * S_SCALE) + MOMENTUM * v_i_L3
			v_i1_FL = -eps_FL * (WD * FL + grad_FL_uns + grad_FL_s * S_SCALE) + MOMENTUM * v_i_FL
			
			F1 += v_i1_L1
			F2 += v_i1_L2
			F3 += v_i1_L3
			FL += v_i1_FL
			
			v_i_L1 = v_i1_L1
			v_i_L2 = v_i1_L2
			v_i_L3 = v_i1_L3
			v_i_FL = v_i1_FL
			
			################
			# diagnostics
			grad_L1_uns *= eps_F1
			grad_L2_uns *= eps_F2
			grad_L3_uns *= eps_F3
			grad_FL_uns *= eps_FL
			
			grad_L1_s *= eps_F1 * S_SCALE
			grad_L2_s *= eps_F2 * S_SCALE
			grad_L3_s *= eps_F3 * S_SCALE
			grad_FL_s *= eps_FL * S_SCALE
			
			#######################################
			savemat(filename, {'F1': F1, 'F2': F2, 'F3':F3, 'FL': FL, 'eps_FL': eps_FL, 'eps_F3': eps_F3, 'eps_F2': eps_F2, 'step': step, 'eps_F1': eps_F1, 'N_IMGS': N_IMGS, 'N_TEST_IMGS': N_TEST_IMGS,'err_test':err_test,'err':err,'class_err':class_err,'class_err_test':class_err_test,'epoch_err':epoch_err})
			print iter, batch, step, err_test[-1], class_err_test[-1], t_grad_start, t_grad_s_start, time.time() - t_total, filename
			print '                        F1', np.mean(np.abs(v_i1_L1))/np.mean(np.abs(F1)), 'F2', np.mean(np.abs(v_i1_L2))/np.mean(np.abs(F2)), 'F3', np.mean(np.abs(v_i1_L3))/np.mean(np.abs(F3)), 'FL', np.mean(np.abs(v_i1_FL))/np.mean(np.abs(FL))
			print '                        F1', np.mean(np.abs(grad_L1_uns))/np.mean(np.abs(F1)), 'F2', np.mean(np.abs(grad_L2_uns))/np.mean(np.abs(F2)), 'F3', np.mean(np.abs(grad_L3_uns))/np.mean(np.abs(F3)), 'FL', np.mean(np.abs(grad_FL_uns))/np.mean(np.abs(FL)), ' uns'
			print '                        F1', np.mean(np.abs(grad_L1_s))/np.mean(np.abs(F1)), 'F2', np.mean(np.abs(grad_L2_s))/np.mean(np.abs(F2)), 'F3', np.mean(np.abs(grad_L3_s))/np.mean(np.abs(F3)), 'FL', np.mean(np.abs(grad_FL_s))/np.mean(np.abs(FL)), ' s'
			epoch_err_t += err[-1]
	epoch_err.append(epoch_err_t)
	print '------------ epoch err ----------'
	print epoch_err_t

