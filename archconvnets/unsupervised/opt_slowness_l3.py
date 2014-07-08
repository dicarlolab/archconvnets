from scipy.spatial.distance import squareform
import copy
import time
from scipy.stats import pearsonr
import scipy.optimize
import numpy as np
import random
from scipy.spatial.distance import pdist
from scipy.io import loadmat
from scipy.io import savemat
from scipy.stats.mstats import zscore
import math
from scipy.signal import convolve2d
from archconvnets.unsupervised.conv import conv_block

def max_pool_locs(conv_output, crop_derivs=True):
	# conv_output: n_filters, output_sz, output_sz, n_imgs
	# output_deriv: in_channels, filter_sz, filter_sz, output_sz, output_sz, n_imgs
	assert len(conv_output.shape) == 4
	assert conv_output.shape[1] == conv_output.shape[2]
	output = np.zeros((n_filters, output_sz2, output_sz2, n_imgs),dtype='float32')
	if crop_derivs == True:
		output_deriv_max = np.zeros((in_channels, filter_sz, filter_sz, n_filters, output_sz2, output_sz2, n_imgs),dtype='float32')
	x_loc = 0
	for x in range(output_sz2):
		y_loc = 0
		for y in range(output_sz2):
			for filter in range(n_filters):
				output_patch = conv_output[filter,x_loc:x_loc+pool_window_sz, y_loc:y_loc+pool_window_sz]
				if crop_derivs == True:
					deriv_patch = output_deriv[:,:,:,x_loc:x_loc+pool_window_sz, y_loc:y_loc+pool_window_sz]
				
				output_patch = output_patch.reshape((pool_window_sz**2, n_imgs))
				if crop_derivs == True:
					deriv_patch = deriv_patch.reshape((in_channels, filter_sz, filter_sz, pool_window_sz**2, n_imgs))
				
				inds = np.argmax(output_patch,axis=0)
				output[filter,x,y] = output_patch[inds, range(n_imgs)]
				if crop_derivs == True: 
					output_deriv_max[:,:,:,filter,x,y] = deriv_patch[:,:,:,inds, range(n_imgs)]
				
			y_loc += pool_stride
		x_loc += pool_stride
	if crop_derivs == True:
		return output, output_deriv_max
	else:
		return output

def test_grad_grad(x):
	global transpose_norm
	global l2_norm
	x = np.float32(x)
	t_start = time.time()
	filters = x.reshape((in_channels, filter_sz, filter_sz, n_filters))
	
	conv_out = np.single(conv_block(np.double(filters), np.double(imgs), stride))
	output, output_deriv_max = max_pool_locs(conv_out)
	
	diffs = np.zeros((n_filters, output_sz2, output_sz2, n_imgs-1),dtype='float32')
	diffs_deriv = np.zeros((in_channels, filter_sz, filter_sz, n_filters, output_sz2, output_sz2, n_imgs-1),dtype='float32')
	for img in range(0, n_imgs-1):
		if ((img+1) % frames_per_movie) != 0: # skip movie boundaries
			diffs[:,:,:,img] = output[:,:,:,img] - output[:,:,:,img+1]
			diffs_deriv[:,:,:,:,:,:,img] = output_deriv_max[:,:,:,:,:,:,img] - output_deriv_max[:,:,:,:,:,:,img+1]
	sign_mat = (1 - 2*(diffs < 0)).reshape((1,1,1,n_filters, (output_sz2**2)*(n_imgs-1)))
	diffs = diffs.reshape((1,1,1,n_filters, (output_sz2**2)*(n_imgs-1)))
	diffs_deriv = diffs_deriv.reshape((in_channels, filter_sz, filter_sz, n_filters, (output_sz2**2)*(n_imgs-1)))
	grad_diffs = 2*np.sum(diffs_deriv*diffs, axis=4).ravel()
	
	loss_diffs = np.sum(diffs**2)
	
	x_back = copy.deepcopy(x)
	########## transpose
	x_in = copy.deepcopy(x)
	x = np.reshape(x, (in_channels*(filter_sz**2), n_filters)).T
	
	corrs = (1-pdist(x,'correlation'))
	loss_t = np.sum(np.abs(corrs))
	corr_mat = squareform(corrs)
	
	grad_t = np.zeros((in_channels*(filter_sz**2), n_filters),dtype='float32').T
	
	d = x - np.mean(x,axis=1)[:,np.newaxis]
	d_sum_n = np.sum(d, axis=1) / (in_channels*(filter_sz**2))
	d2_sum_sqrt = np.sqrt(np.sum(d**2, axis=1))
	d2_sum_sqrt2 = d2_sum_sqrt**2
	d_minus_sum_n = d - d_sum_n[:,np.newaxis]
	d_minus_sum_n_div = d_minus_sum_n/d2_sum_sqrt[:,np.newaxis]
	d_dot_dT = np.dot(d, d.T)
	
	sign_mat = np.ones((n_filters, n_filters)) - 2*(corr_mat < 0)

	for i in np.arange(n_filters):
		for j in np.arange(n_filters):
			if i != j:
				grad_t[i] += sign_mat[i,j]*(d_minus_sum_n[j]*d2_sum_sqrt[i] - d_dot_dT[i,j]*d_minus_sum_n_div[i])/(d2_sum_sqrt[j]*d2_sum_sqrt2[i])             
	grad_t = grad_t.T.ravel()
	
	######## l2
	x = x_back
	x = x.ravel()
	loss_l2 = np.sum(x**2)
	grad_l2 = 2*x
	
	#if loss_l2 > 2500:
	#	loss_l2 = 2500
	#	grad_l2 = 0
	
	'''loss_l2 = np.sum(np.abs(x))
	grad_l2 = 1 - 2*(x < 0)
	
	if loss_l2 > 700:
		loss_l2 = 700;
		grad_l2 = 0'''
	
	if transpose_norm == np.inf:
		#transpose_norm = 0.01*(loss_diffs / loss_l2) / loss_t
		transpose_norm = 0.0001 * loss_diffs / loss_t
	if l2_norm == np.inf:
		l2_norm = 0.001* loss_diffs / loss_l2
	#grad = (grad_diffs*loss_l2 - loss_diffs*grad_l2)/(loss_l2**2) + transpose_norm*grad_t
	#loss = (loss_diffs / loss_l2) + transpose_norm*loss_t
	
	#grad = grad_diffs - l2_norm*grad_l2 + transpose_norm*grad_t
	#loss = loss_diffs - l2_norm*loss_l2 + transpose_norm*loss_t
	
	grad = grad_diffs - l2_norm*grad_l2*(1/(loss_l2**2)) + transpose_norm*grad_t
	loss = loss_diffs + l2_norm*(1/loss_l2) + transpose_norm*loss_t
	
	if np.isnan(loss) == False:
		savemat('slowness_filters_l3.mat', {'filters':filters})
	else:
		print 'nan, not saved'
	print loss, loss_diffs, loss_t, loss_l2, time.time() - t_start
	return np.double(loss), np.double(grad)
	#return loss_t, grad_t

#################
# load images
padding = 2
n_batches_load = 6#32#16
img_sz = 138
n_imgs = n_batches_load * 128
in_channels = 1
imgs = np.zeros((in_channels, img_sz+padding*2, img_sz+padding*2, n_batches_load*128),dtype='float32')
frame_step = 2
frames_per_movie = 150 / frame_step
base_batch = 20000+20+7

data_mean = loadmat('movie_mean.mat')['data_mean']
for batch in range(base_batch, base_batch+n_batches_load):
	x = np.load('batch128_img138_full/data_batch_' + str(batch))['data'] - data_mean.T
	x = x.reshape((3, img_sz, img_sz, 128))[:in_channels]
	imgs[:, padding:img_sz+padding, padding:img_sz+padding, (batch-base_batch)*128:(batch-base_batch+1)*128] = copy.deepcopy(x)
imgs = imgs[:,:,:,range(0,n_batches_load*128,frame_step)]
n_imgs = imgs.shape[3]

##########
n_filters = 64
filter_sz = 7
stride = 2
pool_window_sz = 3 
pool_stride = 2

print n_imgs
n_imgs = 2*32
print n_imgs
imgs = imgs[:,:,:,:n_imgs]

output_sz = len(range(0, img_sz + padding*2 - filter_sz + 1, stride))
output_sz2 = len(range(0, output_sz - pool_window_sz + 1, pool_stride))


filters = zscore(loadmat('slowness_filters.mat')['filters'], axis=None)
conv_out = np.single(conv_block(np.double(filters), np.double(imgs), stride))
output = max_pool_locs(conv_out, crop_derivs=False)
img_sz = output.shape[1]

# pad output
img_sz = output.shape[1]
imgs = np.zeros((output.shape[0], img_sz+padding*2, img_sz+padding*2, n_imgs),dtype='float32')
imgs[:, padding:img_sz+padding, padding:img_sz+padding] = copy.deepcopy(output)

############ layer2
filter_sz = 5
stride = 1
in_channels = imgs.shape[0]

output_sz = len(range(0, img_sz + padding*2 - filter_sz + 1, stride))
output_sz2 = len(range(0, output_sz - pool_window_sz + 1, pool_stride))


filters = zscore(loadmat('slowness_filters_l2.mat')['filters'], axis=None)
conv_out = np.single(conv_block(np.double(filters), np.double(imgs), stride))
output = max_pool_locs(conv_out, crop_derivs=False)
img_sz = output.shape[1]

# pad output
img_sz = output.shape[1]
imgs = np.zeros((output.shape[0], img_sz+padding*2, img_sz+padding*2, n_imgs),dtype='float32')
imgs[:, padding:img_sz+padding, padding:img_sz+padding] = copy.deepcopy(output)

########## layer3
filter_sz = 3
in_channels = imgs.shape[0]

output_sz = len(range(0, img_sz + padding*2 - filter_sz + 1, stride))
output_sz2 = len(range(0, output_sz - pool_window_sz + 1, pool_stride))

######## re-compute conv derivs or not
if True:
	print 'starting deriv convs'
	output_deriv = np.zeros((in_channels, filter_sz, filter_sz, output_sz, output_sz, n_imgs),dtype='float32')
	for filter_i in range(filter_sz):
		for filter_j in range(filter_sz):
			print filter_i, filter_j
			for channel in range(in_channels):
				temp_filter = np.zeros((in_channels, filter_sz, filter_sz,1),dtype='float32')
				temp_filter[channel,filter_i,filter_j] = 1
				output_deriv[channel,filter_i,filter_j] = conv_block(np.double(temp_filter), np.double(imgs), stride)
	print 'finished deriv convs'
	savemat('conv_derivs_l3.mat', {'output_deriv': output_deriv})
	print 'saved'
else:
	output_deriv = loadmat('conv_derivs_l3.mat')['output_deriv']
###
x0 = np.random.random((in_channels*filter_sz*filter_sz*n_filters,1))
x0 -= np.mean(x0)
#x0 *= 10000
#x0 /= np.sum(np.abs(x0))
transpose_norm = np.inf
l2_norm = np.inf

t_start = time.time()
x,f,d = scipy.optimize.fmin_l_bfgs_b(test_grad_grad, x0)
print time.time() - t_start

