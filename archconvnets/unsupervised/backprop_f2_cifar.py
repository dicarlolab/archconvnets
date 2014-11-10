#from procs import *
import random
from scipy.stats import rankdata
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
from scipy.io import savemat
import numexpr as ne
import time
import numpy as np

def pinv(F):
        return np.dot(F.T, np.linalg.inv(np.dot(F,F.T)))

eps = 1e-3
f = open('/home/darren/j','w')
f2 = open('/home/darren/j2','w')
model = unpickle('/home/darren/cifar_checkpoints/color/conv_2layer_linear_max16/ConvNet__2014-11-08_16.59.23/120.5')

w = np.single(loadmat('/home/darren/w_cifar.mat')['w']) # (10,16*6*6)
weight_ind = 5
neuron_ind = 6
gpu = '1'
tmp_model = '/home/darren/cifar_tmp.model'
feature_path_pool = '/tmp/cifar_pool2'
feature_path_conv = '/tmp/cifar_conv2'
errs_test = []; errs = []
class_errs_test = []; class_errs = []
pad = 2
n_imgs = 2500

F = np.single(np.random.random((16,5,5,16)))
#F = np.single( model['model_state']['layers'][weight_ind]['weights'][0]*21000).reshape((16,5,5,16))
w_shape = model['model_state']['layers'][weight_ind]['weights'][0].shape

pool2_cmd = ['python', '/home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py', '-f', tmp_model, '--test-range=1-6', '--train-range=1-6', '--write-features=pool2', '--feature-path=' + feature_path_pool,  '--gpu=' + gpu]

#################### pool 1
pool1 = np.zeros((16,12,12,0),dtype='single')
for batch in range(1,6):
        x = np.load('/home/darren/cifar_checkpoints/color/conv_2layer_linear_max16/features_pool1/data_batch_' + str(batch))
        pool1 = np.concatenate((pool1,x['data'].reshape((10000,16,12,12)).transpose((1,2,3,0))), axis=3)

x_pad = np.zeros((16,12+pad*2,12+pad*2,50000),dtype='single')
x_pad[:,pad:pad+12,pad:pad+12] = pool1

max_pool1_no_conv = np.zeros((16,5,5,16,6,6,n_imgs),dtype='single')

for step in range(1000):
	for batch_grad in [1]:#range(1,6):
		t_start = time.time()
		###################### save weights
		model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0] = F.reshape(w_shape)
		model['model_state']['layers'][weight_ind]['weights'][0] = F.reshape(w_shape)

		pickle(tmp_model, model)

		################### compute pool2, load results
		subprocess.call(pool2_cmd, stdout=f, stderr=f2)

		pool2 = np.zeros((16*6*6,0),dtype='single')
		labels = np.array((),'int')
		
		if (step == 0) and (batch_grad == 1):
			for batch in range(1,6):
		                x = np.load(feature_path_pool + '/data_batch_' + str(batch_grad))
		                pool2 = np.concatenate((pool2,x['data'].reshape((10000,16*6*6)).T), axis=1)
		                labels = np.concatenate((labels,x['labels'].astype(int)[0]),axis=0)

                        l = np.zeros((10,50000),dtype='int')
			l[labels,range(50000)] = 1
			w = np.dot(l, pinv(pool2))

		
		pool2 = np.zeros((16*6*6,0),dtype='single')
                labels = np.array((),'int')

		#for batch in range(1,6):
		x = np.load(feature_path_pool + '/data_batch_' + str(batch_grad))
		pool2 = np.concatenate((pool2,x['data'].reshape((10000,16*6*6)).T), axis=1)
		labels = np.concatenate((labels,x['labels'].astype(int)[0]),axis=0)
		
		l = np.zeros((10,10000),dtype='int')
		l[labels,range(10000)] = 1

		###################### compute test from pool2
		x = np.load(feature_path_pool + '/data_batch_6')
		d_test = x['data'].reshape((10000,16*6*6)).T
		labels_test = x['labels'].astype(int)[0]
		l_test = np.zeros((10,10000),dtype='int')
		l_test[labels_test,range(10000)]=1

		pred = np.dot(w, d_test)
		errs_test.append(np.sum(np.abs(pred - l_test)))
		class_errs_test.append(1-np.sum(labels_test == np.argmax(pred,axis=0))/10000.0)

		###################### compute conv2, load results
		conv2_cmd = ['python', '/home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py', '-f', tmp_model, '--test-range='+str(batch_grad), '--train-range='+str(batch_grad), '--write-features=conv2', '--feature-path=' + feature_path_conv,  '--gpu=' + gpu]
		subprocess.call(conv2_cmd, stdout=f, stderr=f2)

		conv_out = np.zeros((16,12,12,0),dtype='single')
		#for batch in range(1,6):
		conv2d = np.load(feature_path_conv + '/data_batch_' + str(batch_grad))
		conv_out = np.concatenate((conv_out,conv2d['data'].reshape((10000,16,12,12)).transpose((1,2,3,0))),axis=3)

		###################### find pool1 outputs to compute gradient
		t_pool = time.time()
		x_loc_ind = 0
		for x_loc in range(0,12,2):
		    y_loc_ind = 0
		    for y_loc in range(0,12,2):
			t = conv_out[:, x_loc:x_loc+3,y_loc:y_loc+3, :n_imgs]
			t = t.reshape((16,t.shape[1]*t.shape[2],n_imgs))
			argmax_t = t.argmax(1)
			
			for grad_x in range(5):
			    for grad_y in range(5):
				pool1t = x_pad[:, x_loc+grad_x:x_loc+grad_x+3,y_loc+grad_y:y_loc+grad_y+3, (batch_grad-1)*10000:(batch_grad-1)*10000+n_imgs]
				pool1t = pool1t.reshape((16,pool1t.shape[1]*pool1t.shape[2],n_imgs))
				for grad_f in range(16):
					max_pool1_no_conv[:,grad_x,grad_y,grad_f,x_loc_ind,y_loc_ind] = pool1t[:,argmax_t[grad_f],range(n_imgs)]
			y_loc_ind += 1
		    x_loc_ind += 1
		t_pool = time.time() - t_pool
		# F: 16,5,5,16
		# w: 10,16*6*6
		# max_pool1_no_conv: 16,5,5,16,6,6,n_imgs
		# pool2: 16*6*6,n_imgs
		

		######################################################################
		# compute F2 grad
		wt = w.reshape((10,1,1,1,16,6*6,1))
		pred = np.dot(w,pool2)[:,:n_imgs] # 10, n_imgs
		sign_mat = np.single(1 - 2*(pred < l[:,:n_imgs])) # 10, n_imgs
		wt_sign = wt*sign_mat[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
		max_pool1_no_convt = max_pool1_no_conv.reshape((16,5,5,16,6*6,n_imgs))
		
		grad = np.zeros((16,5,5,16),dtype='single')
		g_start = time.time()
		for cat in range(10):
			wt_signt = wt_sign[cat]
			grad += ne.evaluate('wt_signt*max_pool1_no_convt').sum(-2).mean(-1)
		F -= eps*grad
		
		errs.append(np.sum(np.abs(pred - l[:,:n_imgs])))
		class_errs.append(1-np.sum(labels[:n_imgs] == np.argmax(pred,axis=0))/np.single(n_imgs))

		print batch_grad, errs[-1], class_errs[-1],  errs_test[-1], class_errs_test[-1], time.time() - t_start, time.time() - g_start, t_pool


