from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from math import pi, sin, cos
from direct.showbase.ShowBase import ShowBase
from pandac.PandaModules import loadPrcFileData
import PIL
import PIL.Image

SAVE_FREQ = 10

EPS = 1e-3
MOM_WEIGHT = 0.95

IMG_SZ = 92

F1_scale = 1e-2
F2_scale = 1e-2
F3_scale = 1e-2
FL_imgnet_scale = 1e-4

N = 64#32
n1 = N # L1 filters
n2 = N# ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 4 # ...
s1 = 5

N_C = 3 # directions M, L, R

file_name = '/home/darren/imgnet_92_model.mat'

max_output_sz3  = 12

# these should all be different values:
GPU_CUR = 2

# gpu buffer indices
MAX_OUTPUT1 = 0; DF2_DATA = 1; CONV_OUTPUT1 = 2; DPOOL1 = 3
F1_IND = 4; IMGS_PAD = 5; DF1 = 6; F2_IND = 11; 
D_UNPOOL2 = 12;F3_IND = 13; MAX_OUTPUT2 = 14; MAX_OUTPUT3 = 15
CONV_OUTPUT1 = 19; CONV_OUTPUT2 = 20; CONV_OUTPUT3 = 21
DF2 = 25; DPOOL2 = 26; DF3_DATA = 27; DPOOL3 = 28; DF3 = 29; FL_PRED = 30;
FL_IND = 31; PRED = 32; DFL = 33; IMGS_PAD_IMGNET = 34

MAX_OUTPUT1_IMGNET = 35; CONV_OUTPUT1_IMGNET = 36
MAX_OUTPUT2_IMGNET = 37; MAX_OUTPUT3_IMGNET = 38
CONV_OUTPUT1_IMGNET = 39; CONV_OUTPUT2_IMGNET = 40; CONV_OUTPUT3_IMGNET = 41
IMGS_PAD_IMGNET_TEST = 42

MAX_OUTPUT1_IMGNET_TEST = 43; CONV_OUTPUT1_IMGNET_TEST = 44
MAX_OUTPUT2_IMGNET_TEST = 45; MAX_OUTPUT3_IMGNET_TEST = 46
CONV_OUTPUT1_IMGNET_TEST = 47; CONV_OUTPUT2_IMGNET_TEST = 48; CONV_OUTPUT3_IMGNET_TEST = 49

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL_imgnet = np.single(np.random.normal(scale=FL_imgnet_scale, size=(999, n3, max_output_sz3, max_output_sz3)))

set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)

F1_init = copy.deepcopy(F1)

dF1 = np.zeros_like(F1)
dF2 = np.zeros_like(F2)
dF3 = np.zeros_like(F3)
dFL_imgnet = np.zeros_like(FL_imgnet)

dF1_mom = np.zeros_like(F1)
dF2_mom = np.zeros_like(F2)
dF3_mom = np.zeros_like(F3)
dFL_imgnet_mom = np.zeros_like(FL_imgnet)

step = 0
step_imgnet = 0
err_imgnet = 0

err_imgnet_plot = []
err_imgnet_test_plot = []
class_err_imgnet_test = []


#################################### mean image
mean_img = loadmat('/home/darren/archconvnets/archconvnets/unsupervised/reinforcement/mean_img_3d.mat')['mean_img']

##################
# load test imgs into buffers (imgnet).. total: 4687 batches
N_BATCHES_IMGNET = 4687
N_BATCHES_IMGNET_TEST = 5
IMGNET_BATCH_SZ = 256
N_TRAIN_IMGNET = N_BATCHES_IMGNET_TEST*IMGNET_BATCH_SZ

imgnet_batch = N_BATCHES_IMGNET_TEST + 1

imgs_pad_test_imgnet = np.zeros((3, IMG_SZ, IMG_SZ, N_TRAIN_IMGNET),dtype='single')
Y_test_imgnet = np.zeros((N_TRAIN_IMGNET, 999),dtype='single')
labels_test_imgnet = np.zeros(N_TRAIN_IMGNET, dtype='int')

for batch in range(1,N_BATCHES_IMGNET_TEST+1):
	z2 = np.load('/export/storage/imgnet92/data_batch_' + str(batch))
	x2 = z2['data'].reshape((3, 92, 92, IMGNET_BATCH_SZ))

	labels_temp = np.asarray(z2['labels']).astype(int)
	
	Y_test_imgnet[(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ][range(IMGNET_BATCH_SZ),labels_temp] = 1
	labels_test_imgnet[(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ] = copy.deepcopy(labels_temp)

	imgs_pad_test_imgnet[:,:,:,(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ] = x2
imgs_pad_test_imgnet = np.ascontiguousarray(imgs_pad_test_imgnet.transpose((3,0,1,2)))
set_buffer(imgs_pad_test_imgnet - mean_img, IMGS_PAD_IMGNET_TEST, gpu=GPU_CUR)
Y_test_imgnet = Y_test_imgnet.T


#########################################################################

t_start = time.time()

while True:
	s_imgnet = step % 4
	
	if s_imgnet == 0:
		##################
		# load train imgs into buffers
		z = np.load('/export/storage/imgnet92/data_batch_' + str(imgnet_batch))
		
		imgs_pad = np.zeros((IMGNET_BATCH_SZ, 3, IMG_SZ, IMG_SZ),dtype='single')
		Y_train = np.zeros((999, IMGNET_BATCH_SZ), dtype='uint8')

		x2 = z['data'].reshape((3, IMG_SZ, IMG_SZ, IMGNET_BATCH_SZ))

		labels = np.asarray(z['labels'])

		l = np.zeros((IMGNET_BATCH_SZ, 999),dtype='uint8')
		l[np.arange(IMGNET_BATCH_SZ),np.asarray(z['labels']).astype(int)] = 1
		Y_train = l.T

		imgs_pad = np.ascontiguousarray(x2.transpose((3,0,1,2)))
		
		imgnet_batch += 1
		if imgnet_batch > N_BATCHES_IMGNET:
			imgnet_batch = N_BATCHES_IMGNET_TEST + 1
	
	set_buffer(imgs_pad[s_imgnet*64:(s_imgnet+1)*64] - mean_img, IMGS_PAD_IMGNET, gpu=GPU_CUR)
	
	############# forward
	conv_buffers(F1_IND, IMGS_PAD_IMGNET, CONV_OUTPUT1_IMGNET, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT1_IMGNET, MAX_OUTPUT1_IMGNET, gpu=GPU_CUR)
	conv_buffers(F2_IND, MAX_OUTPUT1_IMGNET, CONV_OUTPUT2_IMGNET, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT2_IMGNET, MAX_OUTPUT2_IMGNET, gpu=GPU_CUR)
	conv_buffers(F3_IND, MAX_OUTPUT2_IMGNET, CONV_OUTPUT3_IMGNET, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT3_IMGNET, MAX_OUTPUT3_IMGNET, gpu=GPU_CUR)
	
	max_output3 = return_buffer(MAX_OUTPUT3_IMGNET, gpu=GPU_CUR)
	
	pred = np.einsum(FL_imgnet, range(4), max_output3, [4,1,2,3], [0,4])
	
	pred_m_Y = Y_train[:,s_imgnet*64:(s_imgnet+1)*64] - pred
	
	FL_pred = np.einsum(FL_imgnet, range(4), pred_m_Y, [0,4], [4,1,2,3])
	
	set_buffer(FL_pred, FL_PRED, gpu=GPU_CUR)
	
	########### backward
	max_pool_back_cudnn_buffers(MAX_OUTPUT3_IMGNET, FL_PRED, CONV_OUTPUT3_IMGNET, DPOOL3, gpu=GPU_CUR)
	conv_dfilter_buffers(F3_IND, MAX_OUTPUT2_IMGNET, DPOOL3, DF3, stream=3, gpu=GPU_CUR)
	conv_ddata_buffers(F3_IND, MAX_OUTPUT2_IMGNET, DPOOL3, DF3_DATA, gpu=GPU_CUR)
	max_pool_back_cudnn_buffers(MAX_OUTPUT2_IMGNET, DF3_DATA, CONV_OUTPUT2_IMGNET, DPOOL2, gpu=GPU_CUR)
	conv_ddata_buffers(F2_IND, MAX_OUTPUT1_IMGNET, DPOOL2, DF2_DATA, gpu=GPU_CUR)
	conv_dfilter_buffers(F2_IND, MAX_OUTPUT1_IMGNET, DPOOL2, DF2, stream=2, gpu=GPU_CUR)
	max_pool_back_cudnn_buffers(MAX_OUTPUT1_IMGNET, DF2_DATA, CONV_OUTPUT1_IMGNET, DPOOL1, gpu=GPU_CUR)
	conv_dfilter_buffers(F1_IND, IMGS_PAD_IMGNET, DPOOL1, DF1, stream=1, gpu=GPU_CUR)
	
	
	### return
	dFL_imgnet += np.einsum(max_output3, range(4), pred_m_Y, [4,0], [4,1,2,3])
	dF3 += return_buffer(DF3, stream=3, gpu=GPU_CUR)
	dF2 += return_buffer(DF2, stream=2, gpu=GPU_CUR)
	dF1 += return_buffer(DF1, stream=1, gpu=GPU_CUR)
	
	err_imgnet += np.mean(pred_m_Y**2)
	
	F1 += (dF1 + MOM_WEIGHT*dF1_mom)*EPS / 64
	F2 += (dF2 + MOM_WEIGHT*dF2_mom)*EPS / 64
	F3 += (dF3 + MOM_WEIGHT*dF3_mom)*EPS / 64
	FL_imgnet += (dFL_imgnet + MOM_WEIGHT*dFL_imgnet_mom)*EPS / 64
	
	set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
	set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
	set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)
	
	dF1_mom = copy.deepcopy(dF1)
	dF2_mom = copy.deepcopy(dF2)
	dF3_mom = copy.deepcopy(dF3)
	dFL_imgnet_mom = copy.deepcopy(dFL_imgnet)
	
	dF1 = np.zeros_like(dF1)
	dF2 = np.zeros_like(dF2)
	dF3 = np.zeros_like(dF3)
	dFL_imgnet = np.zeros_like(dFL_imgnet)

	step += 1
	
	if step % SAVE_FREQ == 0:
		###############################################
		# test imgs (imgnet); both action-based and control models
		conv_buffers(F1_IND, IMGS_PAD_IMGNET_TEST, CONV_OUTPUT1_IMGNET_TEST, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT1_IMGNET_TEST, MAX_OUTPUT1_IMGNET_TEST, gpu=GPU_CUR)
		conv_buffers(F2_IND, MAX_OUTPUT1_IMGNET_TEST, CONV_OUTPUT2_IMGNET_TEST, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT2_IMGNET_TEST, MAX_OUTPUT2_IMGNET_TEST, gpu=GPU_CUR)
		conv_buffers(F3_IND, MAX_OUTPUT2_IMGNET_TEST, CONV_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
		max_pool_cudnn_buffers(CONV_OUTPUT3_IMGNET_TEST, MAX_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
		
		max_output3 = return_buffer(MAX_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
		
		pred = np.einsum(FL_imgnet, range(4), max_output3, [4,1,2,3], [0,4])
			
		err_imgnet_test_plot.append(np.mean((pred - Y_test_imgnet)**2))
		class_err_imgnet_test.append(1-(np.argmax(pred,axis=0) == np.asarray(np.squeeze(labels_test_imgnet))).mean())
		
		##
		err_imgnet_plot.append(err_imgnet)
		
		savemat(file_name, {'F1': F1, 'F2': F2, 'F3': F3, 'F1_init': F1_init, 'FL_imgnet': FL_imgnet,\
			'step': step,\
			'err_imgnet_test_plot':err_imgnet_test_plot, 'class_err_imgnet_test':class_err_imgnet_test,\
			'err_imgnet':err_imgnet})
		
		print file_name
		print 'step:', step, 'F1:', np.max(F1), 't:',time.time() - t_start
		print 'err_imgnet:', err_imgnet, 'err_imgnet_test:',err_imgnet_test_plot[-1], \
			'class_imgnet_test:',class_err_imgnet_test[-1]
		
		err_imgnet = 0
		
		t_start = time.time()
