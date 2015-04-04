from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import gnumpy as gpu

def pinv(F):
	return np.dot(np.linalg.inv(np.dot(F.T,F)), F.T)

#kernprof -l bp_movies.py
#python -m line_profiler bp_movies.py.lprof  > p
#@profile
#def sf():

# mcc number of train/test imgs
N_TEST_SET = 1500
N_TRAIN = np.int(N_TEST_SET*.9)
TOP_N = 1

F1_scale = 0.001 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.01

EPS_E = 3
EPS = 2*10**(-EPS_E)

N_IMGS = 100 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34 # input image size (px)
PAD = 2

# compute real BP or patch approx for s31
REAL_BP = True
if REAL_BP == True:
	N_C = 10 # number of categories
	BP_STR = ''
	GPU_SUP = 3
	GPU_UNS = 2
	s_scale = 1
else:
	N_C = 750 # number of patches
	BP_STR = 'patches'
	GPU_SUP = 2
	GPU_UNS = 3
	s_scale = N_IMGS / np.single(N_C)

N = 16
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

file_name = '/home/darren/F1_' + str(N_C) + BP_STR + '_' + str(EPS_E) + 'eps_' + str(N) + 'N.mat'

max_output_sz3  = 5

np.random.seed(6166)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']



# gpu buffer indices
MAX_OUTPUT1 = 0; DF2_DATA = 1; CONV_OUTPUT1 = 2; DPOOL1 = 3
F1_IND = 4; IMGS_PAD = 5; DF1 = 6; F2_IND = 11
D_UNPOOL2 = 12;F3_IND = 13; MAX_OUTPUT2 = 14; MAX_OUTPUT3 = 15
CONV_OUTPUT1 = 19; CONV_OUTPUT2 = 20; CONV_OUTPUT3 = 21
DF2 = 25; DPOOL2 = 26; DF3_DATA = 27; DPOOL3 = 28; DF3 = 29; FL_PRED = 30;
FL_IND = 31; PRED = 32

t_start = time.time()

##################
# load test imgs into buffers
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(6))
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))

labels_test = np.asarray(z['labels'])
l = np.zeros((10000, 10),dtype='int')
l[np.arange(10000),np.asarray(z['labels']).astype(int)] = 1
Y_test = np.single(l.T)

imgs_pad_test = np.zeros((3, IMG_SZ, IMG_SZ, 10000),dtype='single')
imgs_pad_test[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
imgs_pad_test = np.ascontiguousarray(imgs_pad_test.transpose((3,0,1,2)))

##################
# load train imgs into buffers
imgs_pad = np.zeros((6, 10000, 3, IMG_SZ, IMG_SZ),dtype='single')
Y_train = np.zeros((6, 10, 10000), dtype='single')
for batch in range(1,6):
	z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
	x = z['data'] - imgs_mean
	x = x.reshape((3, 32, 32, 10000))

	labels = np.asarray(z['labels'])

	l = np.zeros((10000, 10),dtype='int')
	l[np.arange(10000),np.asarray(z['labels']).astype(int)] = 1
	Y_train[batch] = np.single(l.T)

	imgs_pad[batch,:,:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x.transpose((3,0,1,2))
imgs_pad = np.ascontiguousarray(imgs_pad)

if REAL_BP == False:
	imgs_pads_patch = copy.deepcopy(imgs_pad[batch][:N_C])
	set_buffer(imgs_pads_patch[:N_C], IMGS_PAD, gpu=GPU_SUP)
	
	Ys = np.eye(N_C, dtype='single')

epoch = 0
err = []
class_err = []
mcc_FL = []
mcc_max3 = []


while True:
	t_mcc = time.time()
	
	###############################################
	# test imgs
	conv_output1 = conv(F1, imgs_pad_test[:N_TEST_SET], gpu=GPU_UNS)
	max_output1 = max_pool_cudnn(conv_output1, gpu=GPU_UNS)
	conv_output2 = conv(F2, max_output1, gpu=GPU_UNS)
	max_output2 = max_pool_cudnn(conv_output2, gpu=GPU_UNS)
	conv_output3 = conv(F3, max_output2, gpu=GPU_UNS)
	max_output3 = max_pool_cudnn(conv_output3, gpu=GPU_UNS)
	
	pred = np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4])
	

	#########
	## mcc on FL
	pred_train = pred[:,:N_TRAIN].T
	pred = pred[:,N_TRAIN:N_TEST_SET].T
	
	test_corrs = np.dot(pred, pred_train.T)
	hit = 0
	for test_img in range(N_TEST_SET-N_TRAIN):
		hit += np.max(labels_test[N_TRAIN + test_img] == labels_test[np.argsort(-test_corrs[test_img])[:TOP_N]])
	mcc_FL.append(1-hit/np.single(N_TEST_SET-N_TRAIN))
	
	## mcc on max3
	pred_train = max_output3[:N_TRAIN].reshape((N_TRAIN, n3*max_output_sz3**2))
	pred = max_output3[N_TRAIN:N_TEST_SET].reshape((N_TEST_SET-N_TRAIN, n3*max_output_sz3**2))
	
	test_corrs = np.dot(pred, pred_train.T)
	hit = 0
	for test_img in range(N_TEST_SET-N_TRAIN):
		hit += np.max(labels_test[N_TRAIN + test_img] == labels_test[np.argsort(-test_corrs[test_img])[:TOP_N]])
	mcc_max3.append(1-hit/np.single(N_TEST_SET-N_TRAIN))
	'''
	#########
	## least squares FC
	pred_train = max_output3[:N_TRAIN].reshape((N_TRAIN, n3*max_output_sz3**2))
	pred = max_output3[N_TRAIN:N_TEST_SET].reshape((N_TEST_SET-N_TRAIN, n3*max_output_sz3**2))
	
	w = np.dot(pinv(pred_train), Y_test.T[:N_TRAIN])
	
	pred_remap = np.dot(pred,w)
	err.append(np.mean((pred_remap - Y_test.T[N_TRAIN:N_TEST_SET])**2))
	class_err.append(1-(np.argmax(pred_remap,axis=1) == np.asarray(np.squeeze(labels_test))[N_TRAIN:N_TEST_SET]).mean())
	'''
	#mcc_max3.append(1);mcc_FL.append(1)
	class_err.append(1);err.append(1)
	
	print epoch, 'mccFL:', mcc_FL[-1], 'mccMax3:', mcc_max3[-1], 'LSQclass:', class_err[-1], 'LSQerr:', err[-1], ' F1:', np.sum(np.abs(F1)), time.time() - t_mcc, time.time() - t_start, file_name
	savemat(file_name, {'F1':F1, 'epoch':epoch, 'class_err':class_err, 'err':err,'mcc_FL':mcc_FL, 'mcc_max3':mcc_max3,'F2':F2,'F3':F3,'FL':FL,
		'EPS':EPS,'err':err,'class_err':class_err})
		
	t_start = time.time()
	
	
	for batch in range(1,6):
		for s in range(100):
			grad_F1 = np.zeros_like(F1)
			grad_F2 = np.zeros_like(F2)
			grad_F3 = np.zeros_like(F3)
			
			set_buffer(imgs_pad[batch][s*N_IMGS:(s+1)*N_IMGS], IMGS_PAD, gpu=GPU_UNS)
			
			set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_UNS)
			set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_UNS)
			set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_UNS)
			set_buffer(FL, FL_IND, filter_flag=1, gpu=GPU_UNS)
			
			# forward pass imgs
			conv_buffers(F1_IND, IMGS_PAD, CONV_OUTPUT1, gpu=GPU_UNS)
			max_pool_cudnn_buffers(CONV_OUTPUT1, MAX_OUTPUT1, gpu=GPU_UNS)
			conv_buffers(F2_IND, MAX_OUTPUT1, CONV_OUTPUT2, gpu=GPU_UNS)
			max_pool_cudnn_buffers(CONV_OUTPUT2, MAX_OUTPUT2, gpu=GPU_UNS)
			conv_buffers(F3_IND, MAX_OUTPUT2, CONV_OUTPUT3, gpu=GPU_UNS)
			max_pool_cudnn_buffers(CONV_OUTPUT3, MAX_OUTPUT3, gpu=GPU_UNS)
			
			Ys = Y_train[batch][:,s*N_IMGS:(s+1)*N_IMGS]
			
			######## gradients:
			pred_buffer(FL_IND, MAX_OUTPUT3, PRED, gpu=GPU_UNS) # === np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4])
			pred = return_2d_buffer(PRED, gpu=GPU_UNS)
			
			FL_pred = np.einsum(FL, range(4), pred - Ys, [0,4], [4,1,2,3])
			set_buffer(FL_pred, FL_PRED, gpu=GPU_UNS) # summing across categories
			
			###########

			max_pool_back_cudnn_buffers(MAX_OUTPUT3, FL_PRED, CONV_OUTPUT3, DPOOL3, gpu=GPU_UNS)
			conv_dfilter_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3, stream=3, gpu=GPU_UNS)
			conv_ddata_buffers(F3_IND, MAX_OUTPUT2, DPOOL3, DF3_DATA, gpu=GPU_UNS)
			max_pool_back_cudnn_buffers(MAX_OUTPUT2, DF3_DATA, CONV_OUTPUT2, DPOOL2, gpu=GPU_UNS)
			conv_ddata_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2_DATA, gpu=GPU_UNS)
			conv_dfilter_buffers(F2_IND, MAX_OUTPUT1, DPOOL2, DF2, stream=2, gpu=GPU_UNS)
			max_pool_back_cudnn_buffers(MAX_OUTPUT1, DF2_DATA, CONV_OUTPUT1, DPOOL1, gpu=GPU_UNS)
			conv_dfilter_buffers(F1_IND, IMGS_PAD, DPOOL1, DF1, stream=1, gpu=GPU_UNS)

			###
			max_output3 = return_buffer(MAX_OUTPUT3, gpu=GPU_UNS)
			dFL = np.einsum(max_output3, range(4), pred - Ys, [4,0], [4,1,2,3]) #
			
			dF3 = return_buffer(DF3, stream=3, gpu=GPU_UNS)
			dF2 = return_buffer(DF2, stream=2, gpu=GPU_UNS)
			dF1 = return_buffer(DF1, stream=1, gpu=GPU_UNS)
			
			F1 -= dF1*EPS / N_IMGS
			F2 -= dF2*EPS / N_IMGS
			F3 -= dF3*EPS / N_IMGS
			FL -= dFL*EPS / N_IMGS
		
	epoch += 1
sf()
