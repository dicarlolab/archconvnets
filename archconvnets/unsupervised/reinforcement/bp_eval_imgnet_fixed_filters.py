from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy

file_name = '/export/storage2/reinforcement3d_saves/reinforcement_'

#######################
# load/initialize variables
#z = loadmat(file_name + str(step_load) + '.mat')
z = loadmat(file_name + 'recent.mat')

F1 = np.ascontiguousarray(z['F1'])
F2 = np.ascontiguousarray(z['F2'])
F3 = np.ascontiguousarray(z['F3'])
FL_imgnet = np.ascontiguousarray(z['FL_imgnet'])

n3 = np.int(z['N'])
max_output_sz3 = np.int(z['max_output_sz3'])
FL_imgnet_scale = 1e-3

FL_imgnet = np.single(np.random.normal(scale=FL_imgnet_scale, size=(999, n3, max_output_sz3, max_output_sz3)))

N_BATCHES_IMGNET = np.int(z['N_BATCHES_IMGNET'])
N_BATCHES_IMGNET_TEST = np.int(z['N_BATCHES_IMGNET_TEST'])
IMGNET_BATCH_SZ = np.int(z['IMGNET_BATCH_SZ'])
IMG_SZ = np.int(z['IMG_SZ'])
MOM_WEIGHT = np.single(z['MOM_WEIGHT'])
EPS_IMGNET = 1e-4
SAVE_FREQ = 1
err_imgnet_test_plot = []
err_imgnet_plot = []
class_err_imgnet_test = []
step_imgnet = 0

# these should all be different values:
GPU_CUR = 3

# gpu buffer indices
MAX_OUTPUT1 = 0; DF2_DATA = 1; CONV_OUTPUT1 = 2; DPOOL1 = 3
F1_IND = 4; IMGS_PAD = 5; DF1 = 6; F2_IND = 11; 
D_UNPOOL2 = 12;F3_IND = 13; MAX_OUTPUT2 = 14; MAX_OUTPUT3 = 15
CONV_OUTPUT1 = 19; CONV_OUTPUT2 = 20; CONV_OUTPUT3 = 21
CONVA_OUTPUT1 = 50; CONVA_OUTPUT2 = 51; CONVA_OUTPUT3 = 52
DF2 = 25; DPOOL2 = 26; DF3_DATA = 27; DPOOL3 = 28; DF3 = 29; FL_PRED = 30;
FL_IND = 31; PRED = 32; DFL = 33; IMGS_PAD_IMGNET = 34

MAX_OUTPUT1_IMGNET = 35; CONV_OUTPUT1_IMGNET = 36
MAX_OUTPUT2_IMGNET = 37; MAX_OUTPUT3_IMGNET = 38
CONV_OUTPUT1_IMGNET = 39; CONV_OUTPUT2_IMGNET = 40; CONV_OUTPUT3_IMGNET = 41
IMGS_PAD_IMGNET_TEST = 42

MAX_OUTPUT1_IMGNET_TEST = 43; CONV_OUTPUT1_IMGNET_TEST = 44
MAX_OUTPUT2_IMGNET_TEST = 45; MAX_OUTPUT3_IMGNET_TEST = 46
CONV_OUTPUT1_IMGNET_TEST = 47; CONV_OUTPUT2_IMGNET_TEST = 48; CONV_OUTPUT3_IMGNET_TEST = 49

DA1 = 60; DA2 = 61; DA3 = 62

set_buffer(F1, F1_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F2, F2_IND, filter_flag=1, gpu=GPU_CUR)
set_buffer(F3, F3_IND, filter_flag=1, gpu=GPU_CUR)

dFL_imgnet = np.zeros_like(FL_imgnet)

dFL_imgnet_mom = np.zeros_like(FL_imgnet)

err = 0
err_imgnet = 0


#################################### mean image
mean_img = loadmat('/home/darren/archconvnets/archconvnets/unsupervised/reinforcement/mean_img_3d.mat')['mean_img']

##################
# load test imgs into buffers (imgnet).. total: 4687 batches
N_TRAIN_IMGNET = N_BATCHES_IMGNET_TEST*IMGNET_BATCH_SZ

imgnet_batch = N_BATCHES_IMGNET_TEST + 1

imgs_pad_test_imgnet = np.zeros((3, IMG_SZ, IMG_SZ, N_TRAIN_IMGNET),dtype='single')
Y_test_imgnet = np.zeros((N_TRAIN_IMGNET, 999),dtype='single')
labels_test_imgnet = np.zeros(N_TRAIN_IMGNET, dtype='int')

for batch in range(1,N_BATCHES_IMGNET_TEST+1):
	z2 = np.load('/export/storage/imgnet92/data_batch_' + str(batch))
	x2 = z2['data'].reshape((3, 92, 92, IMGNET_BATCH_SZ))

	labels_temp = np.asarray(z2['labels']).astype(int)
	
	Y_test_imgnet[(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ][range(IMGNET_BATCH_SZ), labels_temp] = 1
	labels_test_imgnet[(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ] = copy.deepcopy(labels_temp)

	imgs_pad_test_imgnet[:,:,:,(batch-1)*IMGNET_BATCH_SZ:batch*IMGNET_BATCH_SZ] = x2
imgs_pad_test_imgnet = np.ascontiguousarray(imgs_pad_test_imgnet.transpose((3,0,1,2)))
set_buffer(imgs_pad_test_imgnet - mean_img, IMGS_PAD_IMGNET_TEST, gpu=GPU_CUR)
Y_test_imgnet = Y_test_imgnet.T
imgnet_loaded = False

###############################################
# test imgs (imgnet); both action-based and control models
conv_buffers(F1_IND, IMGS_PAD_IMGNET_TEST, CONV_OUTPUT1_IMGNET_TEST, gpu=GPU_CUR)
activation_buffers(CONV_OUTPUT1_IMGNET_TEST, CONV_OUTPUT1_IMGNET_TEST, gpu=GPU_CUR)
max_pool_cudnn_buffers(CONV_OUTPUT1_IMGNET_TEST, MAX_OUTPUT1_IMGNET_TEST, gpu=GPU_CUR)

conv_buffers(F2_IND, MAX_OUTPUT1_IMGNET_TEST, CONV_OUTPUT2_IMGNET_TEST, gpu=GPU_CUR)
activation_buffers(CONV_OUTPUT2_IMGNET_TEST, CONV_OUTPUT2_IMGNET_TEST, gpu=GPU_CUR)
max_pool_cudnn_buffers(CONV_OUTPUT2_IMGNET_TEST, MAX_OUTPUT2_IMGNET_TEST, gpu=GPU_CUR)

conv_buffers(F3_IND, MAX_OUTPUT2_IMGNET_TEST, CONV_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
activation_buffers(CONV_OUTPUT3_IMGNET_TEST, CONV_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)
max_pool_cudnn_buffers(CONV_OUTPUT3_IMGNET_TEST, MAX_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)

max_output3_test = return_buffer(MAX_OUTPUT3_IMGNET_TEST, gpu=GPU_CUR)

#########################################################################

t_start = time.time()

while True:
	#########################################################
	# imgnet learning: both action model and control model gradients
	s_imgnet = step_imgnet % 2
	
	if s_imgnet == 0 or imgnet_loaded == False:
		imgnet_loaded = True
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
	
	set_buffer(imgs_pad[s_imgnet*128:(s_imgnet+1)*128] - mean_img, IMGS_PAD_IMGNET, gpu=GPU_CUR)
	
	conv_buffers(F1_IND, IMGS_PAD_IMGNET, CONV_OUTPUT1_IMGNET, gpu=GPU_CUR)
	activation_buffers(CONV_OUTPUT1_IMGNET, CONV_OUTPUT1_IMGNET, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT1_IMGNET, MAX_OUTPUT1_IMGNET, gpu=GPU_CUR)
	
	conv_buffers(F2_IND, MAX_OUTPUT1_IMGNET, CONV_OUTPUT2_IMGNET, gpu=GPU_CUR)
	activation_buffers(CONV_OUTPUT2_IMGNET, CONV_OUTPUT2_IMGNET, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT2_IMGNET, MAX_OUTPUT2_IMGNET, gpu=GPU_CUR)
	
	conv_buffers(F3_IND, MAX_OUTPUT2_IMGNET, CONV_OUTPUT3_IMGNET, gpu=GPU_CUR)
	activation_buffers(CONV_OUTPUT2_IMGNET, CONV_OUTPUT2_IMGNET, gpu=GPU_CUR)
	max_pool_cudnn_buffers(CONV_OUTPUT3_IMGNET, MAX_OUTPUT3_IMGNET, gpu=GPU_CUR)
	
	max_output3 = return_buffer(MAX_OUTPUT3_IMGNET, gpu=GPU_CUR)
	
	pred = np.einsum(FL_imgnet, range(4), max_output3, [4,1,2,3], [0,4])
	
	pred_m_Y = Y_train[:,s_imgnet*128:(s_imgnet+1)*128] - pred
	
	err_imgnet += np.mean(pred_m_Y**2)
	
	dFL_imgnet = np.einsum(max_output3, range(4), pred_m_Y, [4,0], [4,1,2,3])
	
	FL_imgnet += (dFL_imgnet + MOM_WEIGHT*dFL_imgnet_mom)*EPS_IMGNET / 128
	
	dFL_imgnet_mom = copy.deepcopy(dFL_imgnet)
	
	step_imgnet += 1

	if step_imgnet % SAVE_FREQ == 0:
		pred = np.einsum(FL_imgnet, range(4), max_output3_test, [4,1,2,3], [0,4])
			
		err_imgnet_test_plot.append(np.mean((pred - Y_test_imgnet)**2))
		class_err_imgnet_test.append(1-(np.argmax(pred,axis=0) == np.asarray(np.squeeze(labels_test_imgnet))).mean())
	
		##
		err_imgnet_plot.append(err_imgnet)
		
		dic = {'err_imgnet_plot':err_imgnet_plot, 'FL_imgnet': FL_imgnet, \
			'step_imgnet':step_imgnet, 'class_err_imgnet_test':class_err_imgnet_test,'err_imgnet':err_imgnet}
	
		savemat(file_name + 'imgnet_only_smallerF_trans.mat', dic)
		dic = None
	
		print time.time() - t_start, 'err_imgnet:', err_imgnet, 'err_imgnet_test:',err_imgnet_test_plot[-1], \
			'class_imgnet_test:',class_err_imgnet_test[-1]
		print file_name + 'imgnet_only_smallerF_trans.mat'
	
		err_imgnet = 0
		t_start = time.time()
