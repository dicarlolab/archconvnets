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

tmp_model = '/home/darren/tmp_l1.model'
gpu = '0'
feature_path = '/tmp/features'

def DFT_matrix_2d(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    A=np.multiply.outer(i.flatten(), i.flatten())
    B=np.multiply.outer(j.flatten(), j.flatten())
    omega = np.exp(-2*np.pi*1J/N)
    W = np.power(omega, A+B)/N
    return W

def conv_block(filters, base_batch):
	global loss_slow, loss_transpose, loss_fourier
	global tmp_model, gpu, feature_path
	filters = np.single(filters.reshape(weights_shape))
	model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0] = copy.deepcopy(filters)
	model['model_state']['layers'][weight_ind]['weights'][0] = copy.deepcopy(filters)
	model['loss_slow'] = copy.deepcopy(loss_slow)
	model['loss_transpose'] = copy.deepcopy(loss_transpose)
	model['loss_fourier'] = copy.deepcopy(loss_fourier)
	pickle(tmp_model, model)
	f = open('/home/darren/j','w')
	f2 = open('/home/darren/j2','w')
	subprocess.call(['rm', '-r', feature_path])
	subprocess.call(['python', '/home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py', '-f', tmp_model, '--test-range=' + str(np.min(base_batch)) + '-' + str(np.max(base_batch)), '--train-range=0', '--write-features=' + layer_name, '--feature-path=' + feature_path, '--gpu=' + gpu], stdout=f, stderr=f2)
	if len(base_batch) == 1:
		try:
			x = np.load(feature_path + '/data_batch_' + str(base_batch[0]))
			output = x['data'].reshape((n_imgs, n_filters, output_sz, output_sz)).transpose((1,2,3,0))
		except:
			try:
				print 'failed1'
				f = open('/home/darren/j','w')
				f2 = open('/home/darren/j2','w')
				subprocess.call(['rm', '-r', feature_path])
				subprocess.call(['python', '/home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py', '-f', tmp_model, '--test-range=' + str(np.min(base_batch)) + '-' + str(np.max(base_batch)), '--train-range=0', '--write-features=' + layer_name, '--feature-path=' + feature_path, '--gpu=' + gpu], stdout=f, stderr=f2)
				x = np.load(feature_path + '/data_batch_' + str(base_batch[0]))
				output = x['data'].reshape((n_imgs, n_filters, output_sz, output_sz)).transpose((1,2,3,0))
			except:
				try:
					print 'failed2'
					f = open('/home/darren/j','w')
					f2 = open('/home/darren/j2','w')
					subprocess.call(['rm', '-r', feature_path])
        	                        subprocess.call(['python', '/home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py', '-f', tmp_model, '--test-range=' + str(np.min(base_batch)) + '-' + str(np.max(base_batch)), '--train-range=0', '--write-features=' + layer_name, '--feature-path=' + feature_path, '--gpu=' + gpu], stdout=f, stderr=f2)
	                                x = np.load(feature_path + '/data_batch_' + str(base_batch[0]))
					output = x['data'].reshape((n_imgs, n_filters, output_sz, output_sz)).transpose((1,2,3,0))
				except:
					print 'failed3'
		return output
	else:
		return 0

def test_grad_slowness(x):
	global feature_path
	x_in = copy.deepcopy(x)
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
		c = np.load(feature_path + '/data_batch_' + str(batch))
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
	
	
	grad = grad_ldt.ravel()
	
	loss = -corrs_loss
	
	print loss, time.time() - t_start, t_conv, corrs_loss/n_corrs, np.max(x_in)
	return corrs_loss/n_corrs, np.double(grad)
	#return np.double(loss), np.double(grad)
	
def test_grad_fourier(x):
	x_in = copy.deepcopy(x)
	x_shape = x.shape
	x = np.float32(x.reshape((in_channels*(filter_sz**2), n_filters)))
	x = zscore(x,axis=0)
	x = x.reshape(x_shape)
	
	t_start = time.time()
	
	################ fourier
	grad_f = np.zeros((in_channels, sz2, sz2, n_filters))
        Xx_sum = np.zeros(sz2)
        l = 0
        for channel in range(in_channels):
                for filter in range(n_filters):
                        x = x_in.reshape((in_channels, sz2, n_filters))[channel][:,filter]
                        Xx = np.dot(X,x)
                        Xx_sum += Xx
                        l += np.abs(Xx)
                        sign_mat = np.ones_like(Xx) - 2*(Xx < 0)
                        grad_f[channel][:,:,filter] = X * sign_mat[:,np.newaxis]
        sign_mat2 = np.ones(sz2) - 2*(t > l)
        grad_f = (grad_f*sign_mat2[np.newaxis][:,:,np.newaxis,np.newaxis]).sum(1).ravel()
        fourier_loss = np.sum(np.abs(t - l))
	
	#########
	
	grad = grad_f
	loss = fourier_loss
	
	#print loss, fourier_loss, np.max(x_in)
	return np.double(loss), np.double(grad)
	
def test_grad_transpose(x):
	x_in = copy.deepcopy(x)
	x_shape = x.shape
	x = np.float32(x.reshape((in_channels*(filter_sz**2), n_filters)))
	x = zscore(x,axis=0)
	x = x.reshape(x_shape)
	
	t_start = time.time()
	
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
	
	if np.mean(np.abs(corrs)) < 0.15:
		loss_t = 0
		grad_t = 0
	
	loss = loss_t
	grad = grad_t
	
	#print loss, time.time() - t_start, np.mean(np.abs(corrs)), np.max(x_in)
	return np.double(loss), np.double(grad)

#################
# load images
img_sz = 138
n_imgs = 128 # imgs in a batch
in_channels = 1
frames_per_movie = 128#16
base_batches = np.arange(80000, 80000+15)#[80000,80001,80002,80003]

layer_name = 'conv1_1a'
weight_ind = 2
neuron_ind = 3

model = unpickle('/home/darren/movie_64_gray/ConvNet__2014-07-11_19.27.59/30.229')
weights = copy.deepcopy(model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0])
weights_shape = weights.shape

##########
n_filters = 64
filter_sz = 7
sz = filter_sz; sz2 = filter_sz**2

output_sz = 60 

######## re-compute conv derivs or not
if False:
	print 'starting deriv convs'
	output_deriv = np.zeros((in_channels, filter_sz, filter_sz, output_sz, output_sz, n_imgs),dtype='float32')
	for base_batch in base_batches:
		t_start = time.time()
		for filter_i in range(filter_sz):
			for filter_j in range(filter_sz):
				print base_batch, filter_i, filter_j
				for channel in range(in_channels):
					temp_filter = np.zeros((in_channels, filter_sz, filter_sz,n_filters),dtype='float32')
					temp_filter[channel,filter_i,filter_j,0] = 1
					output_deriv[channel,filter_i,filter_j] = conv_block(temp_filter, [base_batch])[0]
		savemat('conv_derivs_' + str(base_batch) + '.mat', {'output_deriv': output_deriv})
		print time.time() - t_start
	print 'finished'
###
x0 = np.random.random((in_channels*filter_sz*filter_sz*n_filters,1))
x0 -= np.mean(x0)
x0 /= np.sum(x0**2)#*(10**10)

####### fourier
X = np.real(DFT_matrix_2d(sz))
t = loadmat('/home/darren/fourier_target.mat')['t'].ravel()

t_start = time.time()
x0 = x0.T
step_sz_slowness = 1e3
step_sz_fourier = 1e1
step_sz_transpose = 1e3

loss_slow = np.zeros(0)
loss_transpose = np.zeros(0)
loss_fourier = np.zeros(0)

for step_g in range(500):
	t_start = time.time()
	
	loss, grad = test_grad_slowness(x0)
	x0 -= step_sz_slowness*grad	
	print 'slowness:', loss
	loss_slow = np.append(loss_slow, loss)

	loss, grad = test_grad_transpose(x0)
	print 'transpose:', loss/2016
	for step in range(200):
		loss, grad = test_grad_transpose(x0)
		x0 -= step_sz_transpose*grad
	loss, grad = test_grad_transpose(x0)
	print loss/2016
	loss_transpose = np.append(loss_transpose, loss/2016)
	
	loss, grad = test_grad_fourier(x0)
        print 'fourier:', loss
        for step in range(15000):
                loss, grad = test_grad_fourier(x0)
                x0 -= step_sz_fourier*grad
        loss, grad = test_grad_fourier(x0)
	print loss
	loss_fourier = np.append(loss_fourier, loss)

	loss, grad = test_grad_transpose(x0)
        print 'transpose: ', loss/2016
	print 'step:', step_g, ' elapsed time:', time.time() - t_start

