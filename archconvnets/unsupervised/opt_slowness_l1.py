from archconvnets.unsupervised.grad_slowness import test_grad_slowness
from procs import *
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

tmp_model = '/export/storage2/tmp_l1.model'
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
	global tmp_model, gpu, feature_path, filters_c
	filters = np.single(filters.reshape(weights_shape))
	model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0] = copy.deepcopy(filters)
	model['model_state']['layers'][weight_ind]['weights'][0] = copy.deepcopy(filters)
	model['loss_slow'] = copy.deepcopy(loss_slow)
	model['loss_transpose'] = copy.deepcopy(loss_transpose)
	model['loss_fourier'] = copy.deepcopy(loss_fourier)
	model['corr_imgnetr'] = copy.deepcopy(corr_imgnetr)
	model['corr_imgnetg'] = copy.deepcopy(corr_imgnetg)
	model['corr_imgnetb'] = copy.deepcopy(corr_imgnetb)
	model['filters'] = copy.deepcopy(filters_c)
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
base_batches = np.arange(80000, 80000+8)

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
step_sz_slowness = 1e-6
step_sz_fourier = 1e1
step_sz_transpose = 1e-5

loss_slow = np.zeros(0)
loss_transpose = np.zeros(0)
loss_fourier = np.zeros(0)
corr_imgnetr = np.zeros(0)
corr_imgnetg = np.zeros(0)
corr_imgnetb = np.zeros(0)

x = unpickle('/home/darren/imgnet_3layer_256_final.model')
rdm_imgnetr = 1-pdist(x['model_state']['layers'][2]['weights'][0][:49],'correlation')
rdm_imgnetg = 1-pdist(x['model_state']['layers'][2]['weights'][0][49:2*49],'correlation')
rdm_imgnetb = 1-pdist(x['model_state']['layers'][2]['weights'][0][49*2:49*3],'correlation')
filters_c = np.zeros((0,x0.shape[1]))

rdm_x = 1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)), 'correlation')
corr_imgnetr = np.append(corr_imgnetr, pearsonr(rdm_x, rdm_imgnetr)[0])
corr_imgnetg = np.append(corr_imgnetg, pearsonr(rdm_x, rdm_imgnetg)[0])
corr_imgnetb = np.append(corr_imgnetb, pearsonr(rdm_x, rdm_imgnetb)[0])
print 'imgnet corrs:', corr_imgnetr[-1]#, corr_imgnetg[-1], corr_imgnetb[-1]

t_loss = np.mean(np.abs(1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)).T, 'correlation')))
print 'transpose:', t_loss

n_cpus = 8
for step_g in range(500):
	t_start = time.time()
	conv_block(x0.reshape((in_channels, filter_sz, filter_sz, n_filters)), base_batches)
	
	l = []
	grad = np.zeros_like(x0)
	for batch in base_batches:
		l.append(proc(test_grad_slowness, feature_path, batch, tmp_model, neuron_ind, in_channels, filter_sz, n_filters, n_imgs, output_sz, frames_per_movie))
		if len(l) == n_cpus:
			print 'computing batch', batch
			results = call(l)
			results = np.asarray(tuple(results))
			grad += results.sum(0)
			l = []
	if len(l) != 0:
		results = call(l)
                results = np.asarray(tuple(results))
                grad += results.sum(0)

	x0 -= step_sz_slowness*grad
	rdm_x = 1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)), 'correlation')
	print 'imgnet corrs:',  pearsonr(rdm_x, rdm_imgnetr)[0]
	
	####################################### transpose
	loss, grad = test_grad_transpose(x0)
	t_loss = np.mean(np.abs(1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)).T, 'correlation')))
	print 'transpose:', t_loss
	for step in range(200):
		loss, grad = test_grad_transpose(x0)
		x0 -= step_sz_transpose*grad
	loss, grad = test_grad_transpose(x0)
	t_loss = np.mean(np.abs(1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)).T, 'correlation')))
	print t_loss
	loss_transpose = np.append(loss_transpose, t_loss)
	
	#loss, grad = test_grad_fourier(x0)
        #print 'fourier:', loss
        #for step in range(15000):
        #        loss, grad = test_grad_fourier(x0)
        #        x0 -= step_sz_fourier*grad
        #loss, grad = test_grad_fourier(x0)
	#print loss
	#loss_fourier = np.append(loss_fourier, loss)

	#loss, grad = test_grad_transpose(x0)
        #print 'transpose: ', loss/2016
	rdm_x = 1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)), 'correlation')
	corr_imgnetr = np.append(corr_imgnetr, pearsonr(rdm_x, rdm_imgnetr)[0])
	filters_c = np.concatenate((filters_c, x0), axis=0)
	
	print 'imgnet corrs:', pearsonr(rdm_x, rdm_imgnetr)[0]
	print 'step:', step_g, ' elapsed time:', time.time() - t_start

