#import numpy as npd
import time
import numpy as np
from archconvnets.unsupervised.conv import conv_block
from archconvnets.unsupervised.pool_inds import max_pool_locs
from archconvnets.unsupervised.compute_L1_grad import L1_grad

eps_F1 = 1e-12
POOL_SZ = 3
POOL_STRIDE = 2
STRIDE1 = 2
N_IMGS = 4
IMG_SZ = 128
N = 4
n1 = N
n2 = N
n3 = N

s3 = 3
s2 = 5
s1 = 5

N_C = 5 # number of categories

output_sz1 = len(range(0, IMG_SZ - s1 + 1, STRIDE1))
max_output_sz1  = len(range(0, output_sz1 - 1, POOL_STRIDE))

output_sz2 = max_output_sz1 - s2 + 1
max_output_sz2  = len(range(0, output_sz2 - 1, POOL_STRIDE))

output_sz3 = max_output_sz2 - s3 + 1
max_output_sz3  = len(range(0, output_sz3 - 1, POOL_STRIDE))

F1 = np.random.random((n1, 3, s1, s1))
F2 = np.random.random((n2, n1, s2, s2))
F3 = np.random.random((n3, n2, s3, s3))
FL = np.random.random((N_C, n3, max_output_sz3, max_output_sz3))
imgs = np.random.random((3, IMG_SZ, IMG_SZ, N_IMGS))
Y = np.random.random((N_C, N_IMGS))


Y_cat_sum = Y.sum(0)
for step in range(10):
	conv_output1 = conv_block(F1.transpose((1,2,3,0)), imgs, stride=STRIDE1)
	max_output1, output_switches1_x, output_switches1_y = max_pool_locs(conv_output1)

	conv_output2 = conv_block(F2.transpose((1,2,3,0)), max_output1)
	max_output2, output_switches2_x, output_switches2_y = max_pool_locs(conv_output2)

	conv_output3 = conv_block(F3.transpose((1,2,3,0)), max_output2)
	max_output3, output_switches3_x, output_switches3_y = max_pool_locs(conv_output3)

	pred = np.dot(FL.reshape((N_C, n3*max_output_sz3**2)), max_output3.reshape((n3*max_output_sz3**2, N_IMGS)))
	err = np.sum((pred - Y)**2)

	pred_cat_sum = pred.sum(0) # sum over categories

	########### F1 deriv wrt f1_, a1_x_, a1_y_, channel_

	t_start = time.time()
	grad = L1_grad(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, pred_cat_sum, Y_cat_sum, imgs, STRIDE1)
	
	F1 -= eps_F1 * grad
	print err, time.time() - t_start

