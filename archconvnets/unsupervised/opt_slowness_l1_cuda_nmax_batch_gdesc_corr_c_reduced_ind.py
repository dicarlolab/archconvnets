import random
import pickle as pk
import time
from scipy.io import savemat
from scipy.stats.mstats import zscore
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
	x = x.reshape((in_channels*(filter_sz**2), n_filters))
	x = zscore(x,axis=0)
	x = x.reshape(x_shape)
	x = np.float32(x)
	t_start = time.time()
	filters = x.reshape((in_channels, filter_sz, filter_sz, n_filters))
	N = in_channels*filter_sz*filter_sz
	t_conv = time.time()
	conv_block(filters, base_batches)
	t_conv = time.time() - t_conv

	loss_gd = 0	
	loss_diffs = 0
	grad_diffs = np.zeros(x.shape)
	
	corrs_r = np.zeros((n_filters, n_imgs-1))
	corrs_loss = 0
	corrs_l = np.zeros((n_filters,n_imgs-1))
	grad_g = np.zeros((in_channels, filter_sz, filter_sz, n_filters))
	grad_gd = np.zeros((in_channels, filter_sz, filter_sz, n_filters))
	grad_gdt = np.zeros((in_channels, filter_sz, filter_sz, n_filters))
	grad_l = np.zeros((in_channels, filter_sz, filter_sz, n_filters))
	grad_ldt = np.zeros((in_channels, filter_sz, filter_sz, n_filters))
	corrs_frames = np.zeros(n_imgs-1)
	corrs_frames_ind = np.zeros((n_filters, n_imgs-1))
	for batch in base_batches:
		print batch
		c = np.load('/tmp/features/data_batch_' + str(batch))
		conv_out = c['data'].reshape((n_imgs, n_filters, output_sz, output_sz)).transpose((1,2,3,0))
		# conv_out: n_filters, output_sz, output_sz, n_imgs
		output_deriv = loadmat('conv_derivs_' + str(batch) + '.mat')['output_deriv'] # in_channels, filter_sz, filter_sz, output_sz, output_sz, n_imgs
		output_deriv = output_deriv.reshape((in_channels, filter_sz, filter_sz, 1, output_sz, output_sz, n_imgs))
		
		conv_out_nmean = conv_out.reshape((n_filters*(output_sz**2), n_imgs))
		conv_out_nmean -= conv_out_nmean.mean(0)
		conv_out_nmean = conv_out_nmean.reshape((1, 1, 1, n_filters, output_sz, output_sz, n_imgs))
		
		# output_deriv*conv_out_nmean: in_channels, filter_sz, filter_sz, n_filters, output_sz, output_sz, n_imgs
		for img in range(0, n_imgs-1):
			if (((img-2) % frames_per_movie) != 0) and (((img+2) % frames_per_movie) != 0) and (((img+1) % frames_per_movie) != 0) and (((img) % frames_per_movie) != 0) and (((img-1) % frames_per_movie) != 0): # skip movie boundaries
				corrs_frames[img] += pearsonr(conv_out[:,:,:,img].ravel(), conv_out[:,:,:,img+1].ravel())[0]
				
				corrs_g = np.sum((conv_out_nmean[:,:,:,:,:,:,img]*conv_out_nmean[:,:,:,:,:,:,img+1]).ravel())
				corrs_gd = np.sqrt(np.sum(conv_out_nmean[:,:,:,:,:,:,img].ravel()**2))*np.sqrt(np.sum(conv_out_nmean[:,:,:,:,:,:,img+1].ravel()**2))
				#corrs_loss += corrs_g / corrs_gd
				grad_g = (output_deriv[:,:,:,:,:,:,img]*conv_out_nmean[:,:,:,:,:,:,img+1]).sum(4).sum(4) # sum over spatial dims
				grad_g += (output_deriv[:,:,:,:,:,:,img+1]*conv_out_nmean[:,:,:,:,:,:,img]).sum(4).sum(4)
				
				grad_gd = (output_deriv[:,:,:,:,:,:,img]*conv_out_nmean[:,:,:,:,:,:,img]).sum(4).sum(4)*np.sqrt((conv_out_nmean[:,:,:,:,:,:,img+1]**2).sum(4).sum(4))/np.sqrt((conv_out_nmean[:,:,:,:,:,:,img]**2).sum(4).sum(4))
				grad_gd += (output_deriv[:,:,:,:,:,:,img+1]*conv_out_nmean[:,:,:,:,:,:,img+1]).sum(4).sum(4)*np.sqrt((conv_out_nmean[:,:,:,:,:,:,img]**2).sum(4).sum(4))/np.sqrt((conv_out_nmean[:,:,:,:,:,:,img+1]**2).sum(4).sum(4))
				grad_gdt += (corrs_g*grad_gd - grad_g*corrs_gd)/(corrs_gd**2)
				for filter in range(n_filters):
					corrs_frames_ind[filter,img] += pearsonr(conv_out[filter,:,:,img].ravel(), conv_out[filter,:,:,img+1].ravel())[0]
					corrs_l = np.sum((conv_out_nmean[:,:,:,filter,:,:,img]*conv_out_nmean[:,:,:,filter,:,:,img+1]).ravel())
					corrs_ld = np.sqrt(np.sum(conv_out_nmean[:,:,:,filter,:,:,img].ravel()**2))*np.sqrt(np.sum(conv_out_nmean[:,:,:,filter,:,:,img+1].ravel()**2))
					corrs_loss += corrs_l / corrs_ld
					grad_l = (output_deriv[:,:,:,0,:,:,img]*conv_out_nmean[:,:,:,filter,:,:,img+1]).sum(3).sum(3)
	                                grad_l += (output_deriv[:,:,:,0,:,:,img+1]*conv_out_nmean[:,:,:,filter,:,:,img]).sum(3).sum(3)
								# ^ summing over spatial dims
					
					grad_ld = (output_deriv[:,:,:,0,:,:,img]*conv_out_nmean[:,:,:,filter,:,:,img]).sum(3).sum(3)*np.sqrt((conv_out_nmean[:,:,:,filter,:,:,img+1]**2).sum(3).sum(3))/np.sqrt((conv_out_nmean[:,:,:,filter,:,:,img]**2).sum(3).sum(3))
					grad_ld += (output_deriv[:,:,:,0,:,:,img+1]*conv_out_nmean[:,:,:,filter,:,:,img+1]).sum(3).sum(3)*np.sqrt((conv_out_nmean[:,:,:,filter,:,:,img]**2).sum(3).sum(3))/np.sqrt((conv_out_nmean[:,:,:,filter,:,:,img+1]**2).sum(3).sum(3))
	                                #print corrs_l.shape, grad_ld.shape, grad_l.shape, corrs_ld.shape
					grad_ldt[:,:,:,filter] += (corrs_l*grad_ld - grad_l*corrs_ld)/(corrs_ld**2)
					corrs_loss += corrs_l / corrs_ld
	#grad_g = -((N-1)/N)*grad_g.ravel()
	#grad_gd = -((N-1)/N)*grad_gd.ravel()
	#loss_g = -np.sum(corrs_g)
	corrs_frames_ind /= len(base_batches)
	corrs_frames /= len(base_batches)
	corrs_l /= len(base_batches)
	#grad_g /= len(base_batches)
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
	
	loss_l2 = np.sum(np.abs(x))
	grad_l2 = 1 - 2*(x < 0)
	
	loss_l2 = np.sum(x**2)
	grad_l2 = 2*x
	
	if transpose_norm == np.inf:
		transpose_norm = 0.1 * (corrs_loss) / loss_t
	#	transpose_norm = 1e8*((loss_g/loss_l2) / loss_t) #0*1e8*loss_diffs / loss_t
	#if l2_norm == np.inf:
	#	l2_norm = 0.01 * loss_diffs / loss_t
	
	#grad = (grad_diffs*loss_l2 - loss_diffs*grad_l2)/(loss_l2**2) #+ transpose_norm*grad_t
	#loss = (loss_diffs / loss_l2) #+ transpose_norm*loss_t
	
	#grad = grad_diffs - (1/l2_norm)*grad_l2/(loss_l2**2)  + transpose_norm*grad_t
	#loss = loss_diffs + 1/(l2_norm*loss_l2) + transpose_norm*loss_t
	
	#grad = grad_diffs - l2_norm*grad_l2 + transpose_norm*grad_t
	#loss = loss_diffs - l2_norm*loss_l2 + transpose_norm*loss_t
	
	#loss = loss_diffs - 0.1*loss_l2/loss_diffs + 0.1*loss_t/loss_diffs
	#grad = grad_diffs - 0.1*(grad_l2*loss_diffs - loss_l2*grad_diffs)/(grad_diffs**2) + 0.1*(grad_t*loss_diffs - loss_t*grad_diffs)/(grad_diffs**2)
	
	#loss = loss_diffs + transpose_norm*loss_t#/loss_diffs
	grad = grad_ldt.ravel() + transpose_norm*grad_t#dt.ravel()#(grad_g*loss_l2 - loss_g*grad_l2)/(loss_l2**2) + transpose_norm*grad_t #10*(grad_t*loss_diffs - loss_t*grad_diffs)/(grad_diffs**2)
	
	loss = -corrs_loss + transpose_norm*loss_t
	
	#if np.isnan(loss) == False:
	#	savemat('slowness_filters_more_imgs5.mat', {'filters':filters})
	#else:
	#	print 'nan, not saved'
	#print loss, loss_diffs/loss_l2, loss_diffs, loss_l2, loss_t/len(corrs), time.time() - t_start, t_conv, np.mean(corrs), np.median(corrs), np.mean(corrs_g), np.median(corrs_g)
	print loss, np.mean(corrs_frames), time.time() - t_start, np.mean(np.abs(corrs)), np.mean(corrs_frames_ind)#np.mean(corrs_l)
	return np.double(loss), np.double(grad)

#################
# load images
img_sz = 138
n_imgs = 128 # imgs in a batch
in_channels = 1
frames_per_movie = 128#16
base_batches = [80000,80001]#,90002,90003]
#base_batches = [90000,90001]
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

