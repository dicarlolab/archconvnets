from archconvnets.unsupervised.conv_block_call import conv_block
from archconvnets.unsupervised.DFT import DFT_matrix_2d
from archconvnets.unsupervised.grad_fourier import test_grad_fourier
from archconvnets.unsupervised.grad_transpose import test_grad_transpose
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
gpu = '1'
feature_path = '/tmp/features'

n_imgs = 128 # imgs in a batch
in_channels = 1
frames_per_movie = 128
base_batches = np.arange(90000, 90000+32)

layer_name = 'conv1_1a'
weight_ind = 2
neuron_ind = 3

model = unpickle('/home/darren/movie_128_gray_5layer/ConvNet__2014-08-08_18.34.10/1.80')
weights = copy.deepcopy(model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0])
weights_shape = weights.shape

##########
n_filters = 128
filter_sz = 7

output_sz = 60 

loss_slow = np.zeros(0)
loss_transpose = np.zeros(0)
loss_fourier = np.zeros(0)
corr_imgnetr = np.zeros(0)
corr_imgnetg = np.zeros(0)
corr_imgnetb = np.zeros(0)

###
x0 = np.random.random((in_channels*filter_sz*filter_sz*n_filters,1))
x0 -= np.mean(x0)
x0 /= np.sum(x0**2)#*(10**10)

####### fourier
X = np.real(DFT_matrix_2d(filter_sz))
t = loadmat('/home/darren/fourier_target.mat')['t'].ravel()

t_start = time.time()
x0 = x0.T
step_sz_slowness = 1e-6
step_sz_fourier = 1e0
step_sz_transpose = 1 #1e-3 #5e-5

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
for step_g in range(3):
	t_start = time.time()
	conv_block(x0.reshape((in_channels, filter_sz, filter_sz, n_filters)), base_batches, loss_slow, loss_transpose, loss_fourier, corr_imgnetr, gpu, tmp_model, feature_path, filters_c, weights_shape, model, neuron_ind, weight_ind, layer_name, output_sz, n_imgs, n_filters)
	
	l = []
	grad = np.zeros_like(x0)
	for batch in base_batches:
		l.append(proc(test_grad_slowness, feature_path, batch, tmp_model, neuron_ind, in_channels, filter_sz, n_filters, n_imgs, output_sz, frames_per_movie, 'conv_derivs_'))
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
	x0 = zscore(x0.reshape((in_channels*(filter_sz**2), n_filters)), axis=0).reshape((1,in_channels*(filter_sz**2)*n_filters))
	rdm_x = 1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)), 'correlation')
	print 'imgnet corrs:',  pearsonr(rdm_x, rdm_imgnetr)[0]
	
	####################################### transpose
	t_loss = np.mean(np.abs(1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)).T, 'correlation')))
	print 'transpose:', t_loss
	for step in range(200):
		loss, grad = test_grad_transpose(x0, in_channels, filter_sz, n_filters)
		x0 -= step_sz_transpose*grad
	t_loss = np.mean(np.abs(1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)).T, 'correlation')))
	print t_loss
	loss_transpose = np.append(loss_transpose, t_loss)

	rdm_x = 1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)), 'correlation')
        corr_imgnetr = np.append(corr_imgnetr, pearsonr(rdm_x, rdm_imgnetr)[0])
        filters_c = np.concatenate((filters_c, x0), axis=0)

        print 'imgnet corrs:', pearsonr(rdm_x, rdm_imgnetr)[0]
		
	################################# fourier
	'''loss, grad = test_grad_fourier(x0, in_channels, filter_sz, n_filters, t, X)
        print 'fourier:', loss
        for step in range(15000):
                loss, grad = test_grad_fourier(x0, in_channels, filter_sz, n_filters, t, X)
                x0 -= step_sz_fourier*grad
        loss, grad = test_grad_fourier(x0, in_channels, filter_sz, n_filters, t, X)
	print loss
	loss_fourier = np.append(loss_fourier, loss)'''
	##################

	rdm_x = 1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)), 'correlation')
	corr_imgnetr = np.append(corr_imgnetr, pearsonr(rdm_x, rdm_imgnetr)[0])
	filters_c = np.concatenate((filters_c, x0), axis=0)
	
	print 'imgnet corrs:', pearsonr(rdm_x, rdm_imgnetr)[0]
	t_loss = np.mean(np.abs(1-pdist(x0.reshape((in_channels*(filter_sz**2), n_filters)).T, 'correlation')))
        print 'transpose:', t_loss
	print 'step:', step_g, ' elapsed time:', time.time() - t_start

