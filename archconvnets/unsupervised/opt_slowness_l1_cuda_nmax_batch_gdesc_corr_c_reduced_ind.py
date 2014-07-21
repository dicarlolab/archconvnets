import random
import pickle as pk
import time
from scipy.io import savemat
import numpy as n
import os
from time import time, asctime, localtime, strftime
from numpy.random import randn, rand
from numpy import s_, dot, tile, zeros, ones, zeros_like, array, ones_like
from util import *
from data import *
from options import *
from math import ceil, floor, sqrt
from data import DataProvider, dp_types
import sys
import shutil
import platform
from os import linesep as NL
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
import subprocess

def conv_block(filters, base_batch):
	filters = np.single(filters.reshape(weights_shape))
	model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0] = copy.deepcopy(filters)
	model['model_state']['layers'][weight_ind]['weights'][0] = copy.deepcopy(filters)
	pickle('/home/darren/tmp_l1.model', model)
	f = open('/home/darren/j','w')
	f2 = open('/home/darren/j2','w')
	subprocess.call(['rm', '-r', '/tmp/features'])
	subprocess.call(['python', '/home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py', '-f', '/home/darren/tmp_l1.model', '--test-range=' + str(np.min(base_batch)) + '-' + str(np.max(base_batch)), '--train-range=0', '--write-features=' + layer_name, '--feature-path=/tmp/features', '--gpu=1'], stdout=f, stderr=f2)
	if len(base_batch) == 1:
		x = np.load('/tmp/features/data_batch_' + str(base_batch[0]))
		output = x['data'].reshape((n_imgs, n_filters, output_sz, output_sz)).transpose((1,2,3,0))
		return output
	else:
		return 0

def test_grad_grad(x):
	global transpose_norm, l2_norm
	x_shape = x.shape
	x = np.float32(x.reshape((in_channels*(filter_sz**2), n_filters)))
	x = zscore(x,axis=0)
	x = x.reshape(x_shape)
	
	t_start = time.time()
	filters = x.reshape((in_channels, filter_sz, filter_sz, n_filters))
	N = in_channels*filter_sz*filter_sz
	
	t_conv = time.time()
	conv_block(filters, base_batches)
	t_conv = time.time() - t_conv

	corrs_loss = 0
	grad_ldt = np.zeros((in_channels, filter_sz**2, n_filters))
	n_corrs = 0
	for batch in base_batches:
		print batch
		c = np.load('/tmp/features/data_batch_' + str(batch))
		conv_out = c['data'].reshape((n_imgs, n_filters, output_sz**2)).transpose((1,2,0))
		# conv_out: n_filters, output_sz**2, n_imgs
		output_deriv = loadmat('conv_derivs_' + str(batch) + '.mat')['output_deriv']
		# output_deriv: in_channels, filter_sz**2, output_sz**2, n_imgs
		output_deriv = output_deriv.reshape((in_channels, filter_sz**2, 1, output_sz**2, n_imgs))
		
		conv_out_nmean = conv_out.reshape((n_filters, output_sz**2, n_imgs))
		conv_out_nmean -= conv_out_nmean.mean(1)[:,np.newaxis]
		conv_out_nmean_std = np.sqrt(np.sum(conv_out_nmean**2,axis=1))
		conv_out_nmean_std_pad = conv_out_nmean_std[np.newaxis, np.newaxis]
		conv_out_nmean_pad = conv_out_nmean[np.newaxis, np.newaxis]
		
		grad_ld1 = (output_deriv*conv_out_nmean_pad).sum(3) / conv_out_nmean_std_pad
		# output_deriv*conv_out_nmean: in_channels, filter_sz**2, n_filters, output_sz**2, n_imgs
		for img in range(0, n_imgs-1):
			if (((img-2) % frames_per_movie) != 0) and (((img+2) % frames_per_movie) != 0) and (((img+1) % frames_per_movie) != 0) and (((img) % frames_per_movie) != 0) and (((img-1) % frames_per_movie) != 0): # skip movie boundaries
				std_pair = conv_out_nmean_std[:,img]*conv_out_nmean_std[:,img+1]
				corrs_l = np.sum(conv_out_nmean[:,:,img]*conv_out_nmean[:,:,img+1],axis=1)

				corrs_loss += np.sum(corrs_l / std_pair)
				grad_l = (output_deriv[:,:,:,:,img]*conv_out_nmean_pad[:,:,:,:,img+1]).sum(3)
				grad_l += (output_deriv[:,:,:,:,img+1]*conv_out_nmean_pad[:,:,:,:,img]).sum(3)
				
				grad_ld = grad_ld1[:,:,:,img]*conv_out_nmean_std_pad[:,:,:,img+1]
				grad_ld += grad_ld1[:,:,:,img+1]*conv_out_nmean_std_pad[:,:,:,img]
				
				grad_ldt += (corrs_l*grad_ld - grad_l*std_pair)/(std_pair**2)
				n_corrs += n_filters
	########## transpose
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
	
	if transpose_norm == np.inf:
		transpose_norm = 0.1 * (corrs_loss) / loss_t
	grad = grad_ldt.ravel() + transpose_norm*grad_t
	
	loss = -corrs_loss + transpose_norm*loss_t
	
	print loss, time.time() - t_start, t_conv, np.mean(np.abs(corrs)), corrs_loss/n_corrs
	return np.double(loss), np.double(grad)

#################
# load images
img_sz = 138
n_imgs = 128 # imgs in a batch
in_channels = 1
frames_per_movie = 128#16
base_batches = np.arange(80000, 80000+3)#[80000,80001,80002,80003]

layer_name = 'conv1_1a'
weight_ind = 2
neuron_ind = 3

model = unpickle('/home/darren/movie_64_gray/ConvNet__2014-07-11_19.27.59/30.229')
weights = copy.deepcopy(model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0])
weights_shape = weights.shape

##########
n_filters = 64
filter_sz = 7

output_sz = 60 

######## re-compute conv derivs or not
if False:#True:
	print 'starting deriv convs'
	output_deriv = np.zeros((in_channels, filter_sz, filter_sz, output_sz, output_sz, n_imgs),dtype='float32')
	for base_batch in base_batches:
		for filter_i in range(filter_sz):
			for filter_j in range(filter_sz):
				print base_batch, filter_i, filter_j
				for channel in range(in_channels):
					temp_filter = np.zeros((in_channels, filter_sz, filter_sz,n_filters),dtype='float32')
					temp_filter[channel,filter_i,filter_j,0] = 1
					output_deriv[channel,filter_i,filter_j] = conv_block(temp_filter, [base_batch])[0]
		savemat('conv_derivs_' + str(base_batch) + '.mat', {'output_deriv': output_deriv})
	print 'finished'
###
x0 = np.random.random((in_channels*filter_sz*filter_sz*n_filters,1))
x0 -= np.mean(x0)
x0 /= np.sum(x0**2)#*(10**10)
transpose_norm = np.inf
l2_norm = np.inf

t_start = time.time()
x0 = x0.T
step_sz = 5e-12#1e-9#1e-10#12
step_sz = 1e-1
#for step in range(10):
#	loss, grad = test_grad_grad(x0)
#	x0 += step_sz*grad
x,f,d = scipy.optimize.fmin_l_bfgs_b(test_grad_grad, x0)
print time.time() - t_start

