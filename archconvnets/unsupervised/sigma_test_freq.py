import time
import numpy as np
from archconvnets.unsupervised.cudnn_module import cudnn_module as cm
import archconvnets.unsupervised.sigma31_layers.sigma31_layers as sigma31_layers
from archconvnets.unsupervised.sigma31_layers.sigma31_layers import max_pool_locs, patch_inds, compute_sigma11_gpu, compute_sigma11_lin_gpu, set_img_from_patches
from scipy.io import savemat, loadmat
from scipy.stats import pearsonr
from scipy.stats import zscore
import random
import copy

N_INDS_KEEP = 1000

conv_block_cuda = cm.conv
F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 10000 # batch size
IMG_SZ_CROP = 32 # input image size (px)
IMG_SZ = 34#70#75# # input image size (px)
img_train_offset = 0
PAD = 2

N = 16
n1 = N # L1 filters
n2 = N
n3 = N

s1 = 5
s2 = 5
s3 = 3

N_C = 10 # number of categories

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE)) + 2*PAD

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

np.random.seed(6666)
inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)
inds_keep = np.unique(inds_keep)
N_INDS_KEEP = len(inds_keep)

#####
np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500

sigma31 = np.zeros((N_C, len(inds_keep)), dtype='single')
sigma11 = np.zeros((len(inds_keep), len(inds_keep)), dtype='single')

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
batch = 1
	
t_start = time.time()
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))

################### compute train err
# load imgs
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_IMGS]

IMG = 0

img = copy.deepcopy(x[:,:,:,IMG].ravel())
random.shuffle(img)
x[:,:,:,IMG] = img.reshape((3,32,32))

'''np.random.seed(143)
for i in range(1,6):
	x[:,:,:,-i] = np.random.normal(size=(3,32,32))
'''
labels = np.asarray(z['labels'])[:N_IMGS].astype(int)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))



# forward pass
conv_output1 = conv_block_cuda(F1, imgs_pad)
max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(np.single(conv_output1),warn=False)

max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv_block_cuda(F2, max_output1)
max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(np.single(conv_output2), PAD=2,warn=False)

max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv_block_cuda(F3, max_output2)
max_output3t, output_switches3_x, output_switches3_y = max_pool_locs(np.single(conv_output3), PAD=2,warn=False)

output_switches2_x -= PAD
output_switches2_y -= PAD

output_switches3_x -= PAD
output_switches3_y -= PAD

patches  = patch_inds(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C, inds_keep, warn=False)

t_start = time.time()
sigma11 = compute_sigma11_gpu(patches[1:]) / (N_IMGS - 1)


img_orig = copy.deepcopy(imgs_pad[IMG])
#imgs_pad = imgs_pad[:2]

EPS=1e-9
p = patches[IMG]
err = []
corr = []
step = 0
while True:
	s11_img = compute_sigma11_gpu(p[np.newaxis][[0,0]]) / 2
	diff = sigma11 - s11_img
	g = -2*(np.dot(sigma11 - s11_img, p) - p*(sigma11[range(N_INDS_KEEP),range(N_INDS_KEEP)] - s11_img[range(N_INDS_KEEP),range(N_INDS_KEEP)]))
	p -= EPS*g
	if step % 5 == 0:
		err.append(np.sum(diff**2))
		corr.append(pearsonr(sigma11.ravel(), s11_img.ravel())[0])
		print err[-1], corr[-1]
		
		######## load img
		
		output_switches3_x_new = output_switches3_x[IMG][np.newaxis]
		output_switches3_y_new = output_switches3_y[IMG][np.newaxis]

		output_switches2_x_new = output_switches2_x[IMG][np.newaxis]
		output_switches2_y_new = output_switches2_y[IMG][np.newaxis]

		output_switches1_x_new = output_switches1_x[IMG][np.newaxis]
		output_switches1_y_new = output_switches1_y[IMG][np.newaxis]

		img_new = imgs_pad[IMG][np.newaxis]
		img_new = set_img_from_patches(output_switches3_x_new, output_switches3_y_new, output_switches2_x_new, output_switches2_y_new, output_switches1_x_new, output_switches1_y_new, s1, s2, s3, img_new, inds_keep, p[np.newaxis], warn=False)
		savemat('/home/darren/img_test_switch_16.mat',{'img_new': img_new, 'img_orig': img_orig, 'imgs_mean': imgs_mean,'err':err,'corr':corr})

		imgs_pad[IMG] = copy.deepcopy(img_new[0])
		
		#################### new indices
		inds_keep = np.random.randint(n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3, size=N_INDS_KEEP)
		
		patches  = patch_inds(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C, inds_keep, warn=False)

		t_start = time.time()
		sigma11 = compute_sigma11_gpu(patches[1:]) / (N_IMGS - 1)
		
		conv_output1 = conv_block_cuda(F1, img_new[[0,0]])
		max_output1t, output_switches1_x_new, output_switches1_y_new = max_pool_locs(np.single(conv_output1),warn=False)

		max_output1 = np.zeros((2, n1, max_output_sz1, max_output_sz1),dtype='single')
		max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

		conv_output2 = conv_block_cuda(F2, max_output1)
		max_output2t, output_switches2_x_new, output_switches2_y_new = max_pool_locs(np.single(conv_output2), PAD=2,warn=False)

		max_output2 = np.zeros((2, n2, max_output_sz2, max_output_sz2),dtype='single')
		max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

		conv_output3 = conv_block_cuda(F3, max_output2)
		max_output3t, output_switches3_x_new, output_switches3_y_new = max_pool_locs(np.single(conv_output3), PAD=2,warn=False)

		output_switches2_x_new -= PAD
		output_switches2_y_new -= PAD

		output_switches3_x_new -= PAD
		output_switches3_y_new -= PAD

		patches  = patch_inds(output_switches3_x_new, output_switches3_y_new, output_switches2_x_new, output_switches2_y_new, output_switches1_x_new, output_switches1_y_new, s1, s2, s3, labels[[0,0]], img_new[[0,0]], N_C, inds_keep, warn=False)
		
		output_switches3_x[IMG] = copy.deepcopy(output_switches3_x_new[IMG])
		output_switches3_y[IMG] = copy.deepcopy(output_switches3_y_new[IMG])
		
		output_switches2_x[IMG] = copy.deepcopy(output_switches2_x_new[IMG])
		output_switches2_y[IMG] = copy.deepcopy(output_switches2_y_new[IMG])
		
		output_switches1_x[IMG] = copy.deepcopy(output_switches1_x_new[IMG])
		output_switches1_y[IMG] = copy.deepcopy(output_switches1_y_new[IMG])
		
		p = patches[IMG]
	step += 1
		
