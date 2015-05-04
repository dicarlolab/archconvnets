from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
import numexpr as ne
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu
import scipy

F1_scale = 1e-7 # std of init normal distribution
F2_scale = 0.000001
F3_scale = 0.000001
FL_scale = 0.000001
CEC_SCALE = 1e-7


POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 6 # batch size
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

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 10 # number of categories

#max_output_sz3  = 23
max_output_sz3  = 18

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, 4, 4)))
FCm = np.single(np.random.normal(scale=FL_scale, size=(n4, n1, 33, 33)))
FCi = np.single(np.random.normal(scale=FL_scale, size=(n4, n1, 33, 33)))
FCo = np.single(np.random.normal(scale=FL_scale, size=(n4, n1, 33, 33)))
FCf = np.single(np.random.normal(scale=FL_scale, size=(n4, n1, 33, 33)))
CEC = np.single(np.random.normal(scale=CEC_SCALE, size=(N_IMGS, n4)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n4)))

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

cat_i = 9
sc = 1*1e3

def f(y):
	#FCm[i_ind, j_ind, k_ind, l_ind] = y
	#FCo[i_ind, j_ind, k_ind, l_ind] = y
	#FCf[i_ind, j_ind, k_ind, l_ind] = y
	F1[i_ind, j_ind, k_ind, l_ind] = y
	#FL[cat_i, j_ind] = y
	
	conv_output1 = conv(F1, imgs_pad, PAD=2)
	FCm_output = np.einsum(FCm, range(4), conv_output1, [4, 1,2,3], [4, 0])
	FCi_output = np.einsum(FCi, range(4), conv_output1, [4, 1,2,3], [4, 0])
	FCo_output = np.einsum(FCo, range(4), conv_output1, [4, 1,2,3], [4, 0])
	FCf_output = np.einsum(FCf, range(4), conv_output1, [4, 1,2,3], [4, 0])
	pred = np.einsum(FL, [0,1], FCo_output*(CEC*FCf_output + FCi_output*FCm_output), [2, 1], [2,0])
	
	err = np.sum((pred[:,cat_i] - sc*Y_test[cat_i])**2) # across imgs
	
	return err
	

def g(y):
	#FL[i_ind, j_ind, k_ind, l_ind] = y
	#FCo[i_ind, j_ind, k_ind, l_ind] = y
	#FCf[i_ind, j_ind, k_ind, l_ind] = y
	#FCm[i_ind, j_ind, k_ind, l_ind] = y
	F1[i_ind, j_ind, k_ind, l_ind] = y
	#FL[cat_i, j_ind] = y
	
	conv_output1 = conv(F1, imgs_pad, PAD=2)
	FCm_output = np.einsum(FCm, range(4), conv_output1, [4, 1,2,3], [4, 0])
	FCi_output = np.einsum(FCi, range(4), conv_output1, [4, 1,2,3], [4, 0])
	FCo_output = np.einsum(FCo, range(4), conv_output1, [4, 1,2,3], [4, 0])
	FCf_output = np.einsum(FCf, range(4), conv_output1, [4, 1,2,3], [4, 0])
	pred = np.einsum(FL, [0,1], FCo_output*(CEC*FCf_output + FCi_output*FCm_output), [2, 1], [2,0])
	
	pred_m_Y = pred[:,cat_i] - sc*Y_test[cat_i]
	
	FL_pred = np.einsum(FL[cat_i], [0], pred_m_Y, [1], [1,0])
	
	#dFL = np.squeeze(np.dot(FC_output.T, pred_m_Y[:,np.newaxis]))
	#return 2*dFL[j_ind]
	
	#dFCf = np.einsum(conv_output1, range(4), FL_pred*FCo_output*CEC, [0,4], [4,1,2,3])
	#return 2*dFCf[i_ind, j_ind, k_ind, l_ind]
	
	#dFCo = np.einsum(conv_output1, range(4), FL_pred*(CEC*FCf_output + FCi_output*FCm_output), [0,4], [4,1,2,3])
	#return 2*dFCo[i_ind, j_ind, k_ind, l_ind]
	
	#dFCm = np.einsum(conv_output1, range(4), FCi_output*FL_pred*FCo_output, [0,4], [4,1,2,3])
	#return 2*dFCm[i_ind, j_ind, k_ind, l_ind]
	
	FLFC_pred = np.einsum(FL_pred*(CEC*FCf_output + FCi_output*FCm_output), [0,1], FCo, [1,2,3,4], [0, 2,3,4])
	FLFC_pred += np.einsum(FCo_output*(CEC*FL_pred), [0,1], FCf, [1,2,3,4], [0, 2,3,4])
	FLFC_pred += np.einsum(FCo_output*(FL_pred*FCm_output), [0,1], FCi, [1,2,3,4], [0, 2,3,4])
	FLFC_pred += np.einsum(FCo_output*(FL_pred*FCi_output), [0,1], FCm, [1,2,3,4], [0, 2,3,4])
	
	dF1 = conv_dfilter(F1, imgs_pad, FLFC_pred, PAD=2,warn=False)
	return 2*dF1[i_ind, j_ind, k_ind, l_ind]
	

	
np.random.seed(np.int64(time.time()))
#eps = np.sqrt(np.finfo(np.float).eps)*1e15
eps = np.sqrt(np.finfo(np.float).eps)*1e13


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	i_ind = np.random.randint(F1.shape[0])
	j_ind = np.random.randint(F1.shape[1])
	k_ind = np.random.randint(F1.shape[2])
	l_ind = np.random.randint(F1.shape[3])
	y = -2e0*F1[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps); print gt, gtx, gtx/gt
	y = -1e1*F1[i_ind,j_ind,k_ind,l_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps); print gt, gtx, gtx/gt
	ratios[sample] = gtx/gt
print ratios.mean(), ratios.std()
