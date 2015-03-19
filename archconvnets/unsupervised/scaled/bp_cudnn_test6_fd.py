from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
import numexpr as ne
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import max_pool_locs
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu
import scipy

F1_scale = 1e-4 # std of init normal distribution
F2_scale = 0.0001
F3_scale = 0.0001
FL_scale = 0.0001


POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 20 # batch size
N_TEST_IMGS = N_IMGS #N_SIGMA_IMGS #128*2
IMG_SZ_CROP = 28 # input image size (px)
IMG_SZ = 32 # input image size (px)
img_train_offset = 2
PAD = 2

N = 2
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 10 # number of categories

max_output_sz3  = 23

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, 4, 4)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']

##################
# load test imgs into buffers
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_TEST_IMGS]

labels = np.asarray(z['labels'])[:N_IMGS]

l = np.zeros((N_TEST_IMGS, N_C),dtype='int')
l[np.arange(N_TEST_IMGS),np.asarray(z['labels'])[:N_TEST_IMGS].astype(int)] = 1
Y_test = np.single(l.T)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_TEST_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP]
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

cat_i = 2

def f(y):
	F1[i_ind, j_ind, k_ind, l_ind] = y
	
	conv_output1 = conv(F1, imgs_pad)
	conv_output2 = conv(F2, conv_output1)
	conv_output3 = conv(F3, conv_output2)
	pred_uns = np.einsum(conv_output3, range(4), FL, [4,1,2,3], [4,0,1,2,3])
	
	err = 0
	for cat_i in range(N_C):
		err += np.sum(((1e15)*Y_test[cat_i,:,np.newaxis,np.newaxis,np.newaxis]-pred_uns[cat_i])**2)
	
	return err
	
def g(y):
	F1[i_ind, j_ind, k_ind, l_ind] = y
	grad = np.zeros_like(F1)
	
	conv_output1 = conv(F1, imgs_pad)
	conv_output2 = conv(F2, conv_output1)
	conv_output3 = conv(F3, conv_output2)
	pred_uns2 = np.einsum(conv_output3, range(4), FL**2, [4,1,2,3], [4,0,1,2,3])
	
	for cat_i in range(N_C):
		dconv_output2 = conv_ddata(F3, conv_output2, pred_uns2[cat_i])
		dconv_output2_1 = conv_ddata(F3, conv_output2, 0*1e15*Y_test[cat_i,:,np.newaxis,np.newaxis,np.newaxis]*FL[cat_i])
		
		dconv_output1 = conv_ddata(F2, conv_output1, dconv_output2)
		dconv_output1_1 = conv_ddata(F2, conv_output1, dconv_output2_1)
		
		dF1_uns = conv_dfilter(F1, imgs_pad, dconv_output1)
		dF1_uns_1 = conv_dfilter(F1, imgs_pad, dconv_output1_1)
		
		grad += dF1_uns - dF1_uns_1
	
	return 2*(grad)[i_ind, j_ind, k_ind, l_ind]

np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e30
#eps = np.sqrt(np.finfo(np.float).eps)*5e7

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	i_ind = 0#np.random.randint(F1.shape[0])
	j_ind = np.random.randint(F1.shape[1])
	k_ind = np.random.randint(F1.shape[2])
	l_ind = np.random.randint(F1.shape[3])
	y = -2e0*F1[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps); print gt, gtx, gtx/gt
	y = -1e1*F1[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps); print gt, gtx, gtx/gt
	ratios[sample] = gtx/gt
print ratios.mean(), ratios.std()
