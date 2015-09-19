import numpy as np
import numexpr as ne
import time
from archconvnets.unsupervised.conv_p import conv_block
from archconvnets.unsupervised.pool_p import pool_block
from scipy.io import savemat
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

n_filters = 48
filter_sz = 7
n_imgs = 128
img_sz = 128
in_channels = 3
stride = 2 #conv
o1 = 31

np.random.seed(6666)
sigma_31 = loadmat('/home/darren/max_patches.mat')['sigma_31']
data_mean = np.load('/storage/batch128_img138_full/batches.meta')['data_mean'][:,np.newaxis]


#F = np.single(np.random.random((in_channels,filter_sz,filter_sz,n_filters)))
#F = np.single(np.random.normal(size=(in_channels,filter_sz,filter_sz,n_filters)))
model = unpickle('/home/darren/imgnet_3layer_48.model')
F1 = model['model_state']['layers'][2]['weights'][0][:,:n_filters].ravel()
random.shuffle(F1)
F1 = F1.reshape((in_channels*filter_sz**2,n_filters)).T
F2 = np.single(np.random.normal(size=(999,n_filters,o1**2)))
eps_F1 = 1e-12
eps_F2 = 1e-7
costs = []
#.1, .001, 1e-6
for step in range(100000):
	for batch in range(1,9000):
		t_startt = time.time()

		x = np.load('/storage/batch128_img138_full/data_batch_' + str(batch))
		imgs = x['data'] - data_mean
		imgs = np.single(imgs.reshape((3,138,138,n_imgs))[:,4:4+128,4:4+128])
		labels = x['labels']

		l = np.zeros((999,n_imgs),dtype='int')
		l[labels,range(n_imgs)] = 1
		
		conv_out = conv_block(F1.T.reshape((in_channels,filter_sz,filter_sz,n_filters)), imgs, stride)
		pool_out, patch_out = pool_block(imgs, conv_out, filter_sz)
		# pool_out: n_filters, o1, o1, n_imgs
		pool_out = pool_out.reshape((n_filters, o1**2, n_imgs))
		# patch_out: in_channels, filter_sz, filter_sz, n_filters, o1, o1, n_imgs
		X = patch_out.reshape((in_channels*filter_sz**2, n_filters, o1**2, n_imgs))

		pred_diff = np.zeros((999,n_imgs),dtype='single')
		for c in range(999):
			pred_diff[c] = (l[c] - F2[c,:,:,np.newaxis]*pool_out.reshape((n_filters,o1**2,n_imgs))).sum(1).sum(0)
		pred_diff = pred_diff.ravel()
		#break
		# deriv. F1
		grad_F1 = np.zeros_like(F1,dtype='single')
		for F in range(n_filters):
			t_start = time.time()
			for K in range(in_channels*filter_sz**2):
				t = np.dot(F2[:,F],X[K,F])
				grad_F1[F,K] -= 2*np.dot(t.ravel(), pred_diff)/n_imgs
			if F % 20 == 0:
				print time.time() - t_start
		F1 -= eps_F1 * grad_F1
		
		# deriv. F2
		pred_diff = pred_diff.reshape((999,n_imgs))
		grad_F2 = np.zeros_like(F2,dtype='single')
		for C in range(999):
			grad_F2[C] -= 2*np.mean(pool_out*pred_diff[np.newaxis,np.newaxis,C],axis=-1)
		F2 -= eps_F2 * grad_F2
		
		costs.append(np.sum(pred_diff**2))
		savemat('f.mat',{'F1':F1,'F2':F2,'eps_F1':eps_F1,'eps_F2':eps_F2,'costs':costs})
		print step, batch, time.time() - t_startt, costs[-1]
		print 'F1',np.min(F1), np.max(F1), eps_F1*np.min(grad_F1), eps_F1*np.max(grad_F1)
		print 'F2',np.min(F2), np.max(F2), eps_F2*np.min(grad_F2), eps_F2*np.max(grad_F2)
	#if step == 2:
	#	break
