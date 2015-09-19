import time
import numpy as np
from archconvnets.unsupervised.pool_inds import max_pool_locs
from archconvnets.unsupervised.cudnn_module.cudnn_module import *
from archconvnets.unsupervised.scaled.compute_sigma31_reduced import s31
import archconvnets.unsupervised.sigma31_layers.sigma31_layers as sigma31_layers
from scipy.io import savemat, loadmat
from scipy.stats import zscore

conv_block_cuda = conv
F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 10000 # batch size
IMG_SZ_CROP = 28 # input image size (px)
IMG_SZ = 32 # input image size (px)
img_train_offset = 2
PAD = 2

N = 8
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

np.random.seed(6666)
F1 = np.single(np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1)))
F2 = np.single(np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2)))
F3 = np.single(np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3)))
FL = np.single(np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3)))

F1 = zscore(F1,axis=None)/500
F2 = zscore(F2,axis=None)/500
F3 = zscore(F3,axis=None)/500
FL = zscore(FL,axis=None)/500

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')


################### compute train err
# load imgs
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
x = x[:,:,:,:N_IMGS]

labels = np.asarray(z['labels'])[:N_IMGS].astype(int)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS),dtype='single')
imgs_pad[:,PAD:PAD+IMG_SZ_CROP,PAD:PAD+IMG_SZ_CROP] = x[:,img_train_offset:img_train_offset+IMG_SZ_CROP,img_train_offset:img_train_offset+IMG_SZ_CROP]
imgs_pad = np.ascontiguousarray(imgs_pad.transpose((3,0,1,2)))

# forward pass
t_forward_start = time.time()
conv_output1 = conv_block_cuda(F1, imgs_pad)
max_output1t, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)

max_output1 = np.zeros((N_IMGS, n1, max_output_sz1, max_output_sz1),dtype='single')
max_output1[:,:,PAD:max_output_sz1-PAD,PAD:max_output_sz1-PAD] = max_output1t

conv_output2 = conv_block_cuda(F2, max_output1)
max_output2t, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)

max_output2 = np.zeros((N_IMGS, n2, max_output_sz2, max_output_sz2),dtype='single')
max_output2[:,:,PAD:max_output_sz2-PAD,PAD:max_output_sz2-PAD] = max_output2t

conv_output3 = conv_block_cuda(F3, max_output2)
max_output3t, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)

output_switches3_x -= PAD
output_switches3_y -= PAD

output_switches2_x -= PAD
output_switches2_y -= PAD

output_switches3_x[output_switches3_x < 0] = 0
output_switches3_y[output_switches3_y < 0] = 0

output_switches2_x[output_switches2_x < 0] = 0
output_switches2_y[output_switches2_y < 0] = 0

output_switches3_x[output_switches3_x >= max_output_sz3] = max_output_sz3-1
output_switches3_y[output_switches3_y >= max_output_sz3] = max_output_sz3-1

output_switches2_x[output_switches2_x >= max_output_sz2] = max_output_sz2-1
output_switches2_y[output_switches2_y >= max_output_sz2] = max_output_sz2-1

##################
t_start = time.time()
sigma31_L1t, sigma31_L2t, sigma31_L3t, sigma31_FLt = sigma31_layers.s31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C)
print time.time() - t_start
savemat('/home/darren/sigma31_8N_10k_c.mat',{'sigma31_L1':sigma31_L1t,'sigma31_L2':sigma31_L2t,'sigma31_L3':sigma31_L3t,'sigma31_FL':sigma31_FLt})

'''t_start = time.time()
sigma31_L1, sigma31_L2, sigma31_L3, sigma31_FL = s31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs_pad, N_C)
print time.time() - t_start

print np.sum(np.isclose(sigma31_L1/N_IMGS, sigma31_L1t)), np.prod(sigma31_L1t.shape)
print np.sum(np.isclose(sigma31_L2/N_IMGS, sigma31_L2t)), np.prod(sigma31_L2t.shape)
print np.sum(np.isclose(sigma31_L3/N_IMGS, sigma31_L3t)), np.prod(sigma31_L3t.shape)
print np.sum(np.isclose(sigma31_FL/N_IMGS, sigma31_FLt)), np.prod(sigma31_FLt.shape)'''


