#import numpy as npd
import time
import numpy as np
from archconvnets.unsupervised.conv import conv_block
from archconvnets.unsupervised.pool_inds import max_pool_locs
from archconvnets.unsupervised.avg_a1.compute_L1_grad import L1_grad
from archconvnets.unsupervised.avg_a1.compute_L2_grad import L2_grad
from archconvnets.unsupervised.avg_a1.compute_L3_grad import L3_grad
from archconvnets.unsupervised.avg_a1.compute_FL_grad import FL_grad
from archconvnets.unsupervised.avg_a1.compute_sigma31 import s31
from scipy.io import savemat, loadmat

F1_scale = 0.01 # std of init normal distribution
F2_scale = 0.01
F3_scale = 0.01
FL_scale = 0.3

POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 1 # layer 1 stride
N_IMGS = 256 # batch size
IMG_SZ = 42 # input image size (px)

N = 16
n1 = N # L1 filters
n2 = N # ...
n3 = N

s3 = 3 # L1 filter size (px)
s2 = 5 # ...
s1 = 5

N_C = 10 # number of categories

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1-POOL_SZ, POOL_STRIDE))

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2-POOL_SZ, POOL_STRIDE))

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3-POOL_SZ, POOL_STRIDE))

np.random.seed(666)
F1 = np.random.normal(scale=F1_scale, size=(n1, 3, s1, s1))
F2 = np.random.normal(scale=F2_scale, size=(n2, n1, s2, s2))
F3 = np.random.normal(scale=F3_scale, size=(n3, n2, s3, s3))
FL = np.random.normal(scale=FL_scale, size=(N_C, n3, max_output_sz3, max_output_sz3))

imgs_mean = np.load('/home/darren/cifar-10-py-colmajor/batches.meta')['data_mean']
z = np.load('/home/darren/cifar-10-py-colmajor/data_batch_1')


################### compute train err
# load imgs
x = z['data'] - imgs_mean
x = x.reshape((3, 32, 32, 10000))
step = 0
x = x[:,:,:,step*N_IMGS:(step+1)*N_IMGS]

l = np.zeros((N_IMGS, N_C),dtype='int')
l[np.arange(N_IMGS),np.asarray(z['labels'])[step*N_IMGS:(step+1)*N_IMGS].astype(int)] = 1
Y = np.double(l.T)

imgs_pad = np.zeros((3, IMG_SZ, IMG_SZ, N_IMGS))
imgs_pad[:,5:5+32,5:5+32] = x


# forward pass
t_forward_start = time.time()
conv_output1 = conv_block(F1.transpose((1,2,3,0)), imgs_pad, stride=STRIDE1)
max_output1, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)

conv_output2 = conv_block(F2.transpose((1,2,3,0)), max_output1)
max_output2, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)

conv_output3 = conv_block(F3.transpose((1,2,3,0)), max_output2)
max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)

##################3
t_start = time.time()
print N_IMGS, imgs_pad.shape
sigma31 = s31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, Y, imgs_pad, N_C)
print time.time() - t_start
