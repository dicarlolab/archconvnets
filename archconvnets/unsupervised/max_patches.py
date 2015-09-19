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

np.random.seed(666)
sigma_31 = np.zeros((999, in_channels*filter_sz*filter_sz, n_filters),dtype='single')
data_mean = np.load('/storage/batch128_img138_full/batches.meta')['data_mean'][:,np.newaxis]


#F = np.single(np.random.random((in_channels,filter_sz,filter_sz,n_filters)))
#F = np.single(np.random.normal(size=(in_channels,filter_sz,filter_sz,n_filters)))
model = unpickle('/home/darren/imgnet_3layer_48.model')
F = model['model_state']['layers'][2]['weights'][0].reshape((in_channels,filter_sz,filter_sz,n_filters))

for batch in range(1,9000):
	t_start = time.time()

	x = np.load('/storage/batch128_img138_full/data_batch_' + str(batch))
	imgs = x['data'] - data_mean
	imgs = np.single(imgs.reshape((3,138,138,n_imgs))[:,4:4+128,4:4+128])
	labels = x['labels']

	conv_out = conv_block(F, imgs, stride)
	pool_out, patch_out = pool_block(imgs, conv_out, filter_sz)
	# patch_out: in_channels, filter_sz, filter_sz, n_filters, o, o, n_imgs
	patch_out = patch_out.mean(-2).mean(-2).reshape((in_channels*filter_sz*filter_sz, n_filters,n_imgs))
	# patch_out: in_channels*filter_sz*filter_sz, n_filters, n_imgs

	for f in range(n_filters):
		sigma_31[labels, :, f] += patch_out[:, f, range(n_imgs)].T
	savemat('/home/darren/max_patches.mat', {'F':F, 'sigma_31':sigma_31,'batch':batch})
	print batch, time.time() - t_start