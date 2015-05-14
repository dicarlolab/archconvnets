from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy

F1_scale = 1e2 # std of init normal distribution
F2_scale = 0.1
F3_scale = 0.1
FL_scale = 0.1


POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 1 # batch size
N_TEST_IMGS = N_IMGS #N_SIGMA_IMGS #128*2
IMG_SZ_CROP = 28 # input image size (px)
IMG_SZ = 32 # input image size (px)
img_train_offset = 2
PAD = 2

N = 2
n1 = N # L1 filters
n2 = N# ...
n3 = N
n4 = 5
n5 = 7

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 10 # number of categories

max_output_sz3  = 18

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, 4, 4)))

Bf = np.single(np.random.normal(scale=1, size=n4))*1e-8

FCf = np.single(np.random.normal(scale=1e-8, size=(n4, n1, 33, 33)))
FCo = np.single(np.random.normal(scale=1e-8, size=(n4, n1, 33, 33)))
FCi = np.single(np.random.normal(scale=1e-8, size=(n4, n1, 33, 33)))
FCm = np.single(np.random.normal(scale=1e-8, size=(n4, n1, 33, 33)))
CEC = np.single(np.random.normal(scale=FL_scale, size=(n4)))

FC2f = np.single(np.random.normal(scale=1, size=(n5, n4)))
FC2o = np.single(np.random.normal(scale=1, size=(n5, n4)))
FC2i = np.single(np.random.normal(scale=1, size=(n5, n4)))
FC2m = np.single(np.random.normal(scale=1, size=(n5, n4)))
CEC2 = np.single(np.random.normal(scale=1, size=(n5)))

FL = np.single(np.random.normal(scale=FL_scale, size=(n5)))

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

sc = 1*1e3

def f(y):
	#FC2m[i_ind, j_ind] = y
	#Bf[i_ind] = y
	F1[i_ind, j_ind, k_ind, l_ind] = y
	#FL[i_ind] = y
	#FCf[i_ind, j_ind, k_ind, l_ind] = y
	#FCi[i_ind, j_ind, k_ind, l_ind] = y
	#FCo[i_ind, j_ind, k_ind, l_ind] = y
	#FCm[i_ind, j_ind, k_ind, l_ind] = y
	
	conv_output1 = conv(F1, imgs_pad, PAD=2)
	
	FCf_output_pre = np.einsum(FCf, range(4), conv_output1, [4, 1,2,3], [0]) + Bf
	FCi_output_pre = np.einsum(FCi, range(4), conv_output1, [4, 1,2,3], [0])
	FCo_output_pre = np.einsum(FCo, range(4), conv_output1, [4, 1,2,3], [0])
	FCm_output_pre = np.einsum(FCm, range(4), conv_output1, [4, 1,2,3], [0])
	
	FCf_output = 1 / (1 + np.exp(-FCf_output_pre)) - 10
	FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
	FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
	FCm_output = FCm_output_pre
	
	FC_output = FCo_output * (FCf_output * CEC + FCi_output * FCm_output)
	
	FC2f_output_pre = np.squeeze(np.dot(FC2f, FC_output[:,np.newaxis]))
	FC2o_output_pre = np.squeeze(np.dot(FC2o, FC_output[:,np.newaxis]))
	FC2i_output_pre = np.squeeze(np.dot(FC2i, FC_output[:,np.newaxis]))
	FC2m_output_pre = np.squeeze(np.dot(FC2m, FC_output[:,np.newaxis]))
	
	FC2f_output = 1 / (1 + np.exp(-FC2f_output_pre))
	FC2o_output = 1 / (1 + np.exp(-FC2o_output_pre))
	FC2i_output = 1 / (1 + np.exp(-FC2i_output_pre))
	FC2m_output = FC2m_output_pre
	
	FC2_output = FC2o_output * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	pred = (FL * FC2_output).sum()
	pred_m_Y = pred - 1
	
	err = pred_m_Y**2
	
	return err

def g(y):
	#Bf[i_ind] = y
	F1[i_ind, j_ind, k_ind, l_ind] = y
	#FC2m[i_ind, j_ind] = y
	#FL[i_ind] = y
	#FCf[i_ind, j_ind, k_ind, l_ind] = y
	#FCi[i_ind, j_ind, k_ind, l_ind] = y
	#FCm[i_ind, j_ind, k_ind, l_ind] = y
	#FCo[i_ind, j_ind, k_ind, l_ind] = y
	
	############ forward
	
	conv_output1 = conv(F1, imgs_pad, PAD=2)
	
	FCf_output_pre = np.einsum(FCf, range(4), conv_output1, [4, 1,2,3], [0]) + Bf
	FCi_output_pre = np.einsum(FCi, range(4), conv_output1, [4, 1,2,3], [0])
	FCo_output_pre = np.einsum(FCo, range(4), conv_output1, [4, 1,2,3], [0])
	FCm_output_pre = np.einsum(FCm, range(4), conv_output1, [4, 1,2,3], [0])
	
	FCf_output = 1 / (1 + np.exp(-FCf_output_pre)) - 10
	FCi_output = 1 / (1 + np.exp(-FCi_output_pre))
	FCo_output = 1 / (1 + np.exp(-FCo_output_pre))
	FCm_output = FCm_output_pre
	
	FC_output = FCo_output * (FCf_output * CEC + FCi_output * FCm_output)
	
	FC2f_output_pre = np.squeeze(np.dot(FC2f, FC_output[:,np.newaxis]))
	FC2o_output_pre = np.squeeze(np.dot(FC2o, FC_output[:,np.newaxis]))
	FC2i_output_pre = np.squeeze(np.dot(FC2i, FC_output[:,np.newaxis]))
	FC2m_output_pre = np.squeeze(np.dot(FC2m, FC_output[:,np.newaxis]))
	
	FC2f_output = 1 / (1 + np.exp(-FC2f_output_pre))
	FC2o_output = 1 / (1 + np.exp(-FC2o_output_pre))
	FC2i_output = 1 / (1 + np.exp(-FC2i_output_pre))
	FC2m_output = FC2m_output_pre
	
	FC2_output = FC2o_output * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	pred = (FL * FC2_output).sum()
	pred_m_Y = 2*(pred - 1)
	
	############### reverse pointwise
	
	FC2f_output_rev = np.exp(FC2f_output_pre)/((np.exp(FC2f_output_pre) + 1)**2)
	FC2o_output_rev = np.exp(FC2o_output_pre)/((np.exp(FC2o_output_pre) + 1)**2)
	FC2i_output_rev = np.exp(FC2i_output_pre)/((np.exp(FC2i_output_pre) + 1)**2)
	FC2m_output_rev = 1
	
	FCf_output_rev = np.exp(FCf_output_pre)/((np.exp(FCf_output_pre) + 1)**2)
	FCi_output_rev = np.exp(FCi_output_pre)/((np.exp(FCi_output_pre) + 1)**2)
	FCo_output_rev = np.exp(FCo_output_pre)/((np.exp(FCo_output_pre) + 1)**2)
	FCm_output_rev = 1

	############ FL
	
	dFL = pred_m_Y * FC2_output
	
	above_w = pred_m_Y * FL
	
	######################### mem 2 gradients:
	
	FC2f_output_rev_sig = above_w * FC2o_output * (FC2f_output_rev * CEC2)
	FC2i_output_rev_sig = above_w * FC2o_output * (FC2i_output_rev * FC2m_output)
	FC2m_output_rev_sig = above_w * FC2o_output * (FC2i_output * FC2m_output_rev)
	FC2o_output_rev_sig = above_w * FC2o_output_rev * (FC2f_output * CEC2 + FC2i_output * FC2m_output)
	
	dFC2f = np.einsum(FC_output, [0], FC2f_output_rev_sig, [1], [1,0]) 
	dFC2i = np.einsum(FC_output, [0], FC2i_output_rev_sig, [1], [1,0]) 
	dFC2m = np.einsum(FC_output, [0], FC2m_output_rev_sig, [1], [1,0]) 
	dFC2o = np.einsum(FC_output, [0], FC2o_output_rev_sig, [1], [1,0]) 
	
	above_w = np.einsum(FC2o, [0,1], FC2o_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2f, [0,1], FC2f_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2i, [0,1], FC2i_output_rev_sig, [0], [1])
	above_w += np.einsum(FC2m, [0,1], FC2m_output_rev_sig, [0], [1])
	
	########################## mem 1 gradients:
	
	FCf_output_rev_sig = above_w * FCo_output * (FCf_output_rev * CEC)
	FCi_output_rev_sig = above_w * FCo_output * (FCi_output_rev * FCm_output)
	FCm_output_rev_sig = above_w * FCo_output * (FCi_output * FCm_output_rev)
	FCo_output_rev_sig = above_w * FCo_output_rev * (FCf_output * CEC + FCi_output * FCm_output)
	
	dBf = FCf_output_rev_sig
	
	dFCf = np.einsum(conv_output1, range(4), FCf_output_rev_sig, [4], [4,1,2,3])
	dFCi = np.einsum(conv_output1, range(4), FCi_output_rev_sig, [4], [4,1,2,3])
	dFCm = np.einsum(conv_output1, range(4), FCm_output_rev_sig, [4], [4,1,2,3])
	dFCo = np.einsum(conv_output1, range(4), FCo_output_rev_sig, [4], [4,1,2,3])
	
	above_w = np.einsum(FCo, range(4), FCo_output_rev_sig, [0], [1,2,3])[np.newaxis]
	above_w += np.einsum(FCi, range(4),FCi_output_rev_sig, [0], [1,2,3])[np.newaxis]
	above_w += np.einsum(FCm, range(4),FCm_output_rev_sig, [0], [1,2,3])[np.newaxis]
	above_w += np.einsum(FCf, range(4),FCf_output_rev_sig, [0], [1,2,3])[np.newaxis]
	
	######################### conv gradients:
	
	dF1 = conv_dfilter(F1, imgs_pad, above_w, PAD=2,warn=False)
	
	#return dFCo[i_ind, j_ind, k_ind, l_ind]
	#return dFCm[i_ind, j_ind, k_ind, l_ind]
	#return dFCf[i_ind, j_ind, k_ind, l_ind]
	#return dFCi[i_ind, j_ind, k_ind, l_ind]
	#return dFL[i_ind]
	return dF1[i_ind, j_ind, k_ind, l_ind]
	#return dFC2m[i_ind, j_ind]
	#return dBf[i_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e10


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	i_ind = np.random.randint(F1.shape[0])
	j_ind = np.random.randint(F1.shape[1])
	k_ind = np.random.randint(F1.shape[2])
	l_ind = np.random.randint(F1.shape[3])
	y = -1e0*F1[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);
	
	'''i_ind = np.random.randint(FC2f.shape[0])
	j_ind = np.random.randint(FC2f.shape[1])
	y = -1e0*FC2f[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps); '''
	
	'''i_ind = np.random.randint(FCf.shape[0])
	j_ind = np.random.randint(FCf.shape[1])
	k_ind = np.random.randint(FCf.shape[2])
	l_ind = np.random.randint(FCf.shape[3])
	y = -1e0*FCf[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);'''
	
	'''i_ind = np.random.randint(FL.shape[0])
	y = -1e0*FL[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);'''
	
	'''i_ind = np.random.randint(Bf.shape[0])
	y = -1e0*Bf[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps);'''
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()

