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
N = 16
n1 = N
n2 = N
n3 = N

s3 = 3
s2 = 5
s1 = 5

N_C = 10 # number of categories

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

# float return_px(int f1, int f2, int f3, int z1, int z2, int a3_x, int a3_y, int a2_x, int a2_y, int a1_x, int a1_y, int channel, int img){
f3 = 0; z1 = 2; z2 = 4; img = 3; f2 = 1; a3_x = 0; a3_y = 0; a2_x = 1; a2_y = 3; a1_x_ = 4; a1_y_ = 2; f1_ = 0; channel_ = 0

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
	'''grad = 0

	# pool3 -> conv3
	for img in range(N_IMGS):
 	 for f3 in range(n3):
	  for z1 in range(max_output_sz3):
	   for z2 in range(max_output_sz3):
            a3_x_global = output_switches3_x[f3, z1, z2, img] + a3_x
	    a3_y_global = output_switches3_y[f3, z1, z2, img] + a3_y
	
	    for a2_x in range(s2):
	     for a2_y in range(s2):
	      for f2 in range(n2):
		# pool2 -> conv2
		a2_x_global = output_switches2_x[f2, a3_x_global, a3_y_global, img] + a2_x
		a2_y_global = output_switches2_y[f2, a3_x_global, a3_y_global, img] + a2_y

		# pool1 -> conv1
		a1_x_global = output_switches1_x[f1_, a2_x_global, a2_y_global, img] * STRIDE1 + a1_x_
		a1_y_global = output_switches1_y[f1_, a2_x_global, a2_y_global, img] * STRIDE1 + a1_y_

		# conv1 -> px
		px = imgs[channel_, a1_x_global, a1_y_global, img]

		temp_F_prod_all = F3[f3, f2, a3_x, a3_y] * F2[f2, f1_, a2_x, a2_y] * px

		# supervised term:
		grad -= temp_F_prod_all * Y_cat_sum[img]

		# unsupervised term:
		grad +=  temp_F_prod_all * pred_cat_sum[img];

	F1[f1_, channel_, a1_x_, a1_y_] -= eps_F1 * grad'''

	grad = L1_grad(F1, F2, F3, output_switches3_x.astype('int'), output_switches3_y.astype('int'), output_switches2_x.astype('int'), output_switches2_y.astype('int'), output_switches1_x.astype('int'), output_switches1_y.astype('int'), s1, s2, s3, pred_cat_sum, Y_cat_sum, imgs, STRIDE1)
	
	F1 -= eps_F1 * grad
	print err, time.time() - t_start

