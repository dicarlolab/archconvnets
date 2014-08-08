from archconvnets.unsupervised.conv_block_call import conv_block
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

tmp_model = '/export/storage2/tmp_l3.model'
gpu = '1'
feature_path = '/tmp/features_l3'

n_imgs = 128 # imgs in a batch
in_channels = 64
frames_per_movie = 128
base_batches = np.arange(80000+8*4+8*1, 80000+8*4+8*1+8*1)

layer_name = 'conv3_7a'
weight_ind = 8
neuron_ind = 9

model = unpickle('/export/storage2/tmp_l2_test.model')
weights = copy.deepcopy(model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0])
weights_shape = weights.shape

##########
n_filters = 64
filter_sz = 3

output_sz = 17

######## re-compute conv derivs or not
print 'starting deriv convs'
output_deriv = np.zeros((in_channels, filter_sz, filter_sz, output_sz, output_sz, n_imgs),dtype='float32')
temp_filter = np.zeros((in_channels, filter_sz, filter_sz,n_filters),dtype='float32')
filter_ind = 0
channel_inds = np.zeros(n_filters)
filter_i_inds = np.zeros(n_filters)
filter_j_inds = np.zeros(n_filters)
for base_batch in base_batches:
	t_start = time.time()
	for filter_i in range(filter_sz):
		for filter_j in range(filter_sz):
			print base_batch, filter_i, filter_j
			for channel in range(in_channels):
				temp_filter[channel,filter_i,filter_j,filter_ind] = 1
				channel_inds[filter_ind] = channel
				filter_i_inds[filter_ind] = filter_i
				filter_j_inds[filter_ind] = filter_j
				filter_ind += 1
				if filter_ind == n_filters:
					conv_out = conv_block(temp_filter, [base_batch], [],
                                                [], [], [], gpu, tmp_model, feature_path, [], weights_shape, model,
                                                neuron_ind, weight_ind, layer_name, output_sz, n_imgs, n_filters)
					for i in range(n_filters):
						output_deriv[channel_inds[i],filter_i_inds[i],filter_j_inds[i]] = conv_out[i]
					temp_filter = np.zeros((in_channels, filter_sz, filter_sz,n_filters),dtype='float32')
					filter_ind = 0
	if filter_ind != 0:
		conv_out = conv_block(temp_filter, [base_batch], [],
			[], [], [], gpu, tmp_model, feature_path, [], weights_shape, model,
			neuron_ind, weight_ind, layer_name, output_sz, n_imgs, n_filters)
		for i in range(n_filters):
			output_deriv[channel_inds[i],filter_i_inds[i],filter_j_inds[i]] = conv_out[i]
		temp_filter = np.zeros((in_channels, filter_sz, filter_sz,n_filters),dtype='float32')
		filter_ind = 0
	savemat('conv_derivs_l3_' + str(base_batch) + '.mat', {'output_deriv': output_deriv})
	print time.time() - t_start

