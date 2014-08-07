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

tmp_model = '/export/storage2/tmp_l1.model'
gpu = '0'
feature_path = '/tmp/features'

#################
# load images
img_sz = 138
n_imgs = 128 # imgs in a batch
in_channels = 1
frames_per_movie = 128
base_batches = np.arange(80000, 80000+8*4)

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
				output_deriv[channel,filter_i,filter_j] = conv_block(temp_filter, [base_batch], [], [], [], [], gpu, tmp_model, feature_path, [], weights_shape, model, neuron_ind, weight_ind, layer_name, output_sz, n_imgs, n_filters)[0]
	savemat('conv_derivs_' + str(base_batch) + '.mat', {'output_deriv': output_deriv})
	print time.time() - t_start

