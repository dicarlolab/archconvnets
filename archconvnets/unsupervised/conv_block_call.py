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

def conv_block(filters, base_batch, loss_slow, loss_transpose, loss_fourier, corr_imgnetr, gpu, tmp_model, feature_path, filters_c, weights_shape, model, neuron_ind, weight_ind, layer_name, output_sz, n_imgs, n_filters):
        filters = np.single(filters.reshape(weights_shape))
        model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0] = copy.deepcopy(filters)
        model['model_state']['layers'][weight_ind]['weights'][0] = copy.deepcopy(filters)
        model['loss_slow'] = copy.deepcopy(loss_slow)
        model['loss_transpose'] = copy.deepcopy(loss_transpose)
        model['loss_fourier'] = copy.deepcopy(loss_fourier)
        model['corr_imgnetr'] = copy.deepcopy(corr_imgnetr)
        model['filters'] = copy.deepcopy(filters_c)
        pickle(tmp_model, model)
        f = open('/home/darren/j','w')
        f2 = open('/home/darren/j2','w')
        subprocess.call(['rm', '-r', feature_path])
        cmd = ['python', '/home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py', '-f', tmp_model, '--test-range=' + str(np.min(base_batch)) + '-' + str(np.max(base_batch)), '--train-range=0', '--write-features=' + layer_name, '--feature-path=' + feature_path, '--gpu=' + gpu]
	print cmd
	subprocess.call(cmd, stdout=f, stderr=f2)
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
                                subprocess.call(cmd, stdout=f, stderr=f2)
				x = np.load(feature_path + '/data_batch_' + str(base_batch[0]))
                                output = x['data'].reshape((n_imgs, n_filters, output_sz, output_sz)).transpose((1,2,3,0))
                        except:
                                try:
                                        print 'failed2'
                                        f = open('/home/darren/j','w')
                                        f2 = open('/home/darren/j2','w')
                                        subprocess.call(['rm', '-r', feature_path])
                                        subprocess.call(cmd, stdout=f, stderr=f2)
					x = np.load(feature_path + '/data_batch_' + str(base_batch[0]))
                                        output = x['data'].reshape((n_imgs, n_filters, output_sz, output_sz)).transpose((1,2,3,0))
                                except:
                                        print 'failed3'
                return output
        else:
                return 0
