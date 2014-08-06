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

def test_grad_slowness(feature_path, batch, tmp_model, neuron_ind, in_channels, filter_sz, n_filters, n_imgs, output_sz, frames_per_movie):
        model = unpickle(tmp_model)
        x = copy.deepcopy(model['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0])

        x = np.float32(x.reshape((in_channels*(filter_sz**2), n_filters)))
        x = zscore(x,axis=0)

        t_start = time.time()
        filters = x.reshape((in_channels, filter_sz, filter_sz, n_filters))

        corrs_loss = 0
        grad_ldt = np.zeros((in_channels, filter_sz**2, n_filters))
        n_corrs = 0
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

        # we want to do this, but it takes too much memory, so we need to slice through:
        #grad_ld1 = (output_deriv*conv_out_nmean_pad).sum(3) / conv_out_nmean_std_pad
        grad_ld1 = np.zeros((in_channels, filter_sz**2, n_filters, n_imgs),dtype='float32')
        for img in range(n_imgs):
		grad_ld1[:,:,:,img] = (output_deriv[:,:,:,:,img]*conv_out_nmean_pad[:,:,:,:,img]).sum(3) / conv_out_nmean_std_pad[:,:,:,img]
	
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

        print corrs_loss/n_corrs, time.time() - t_start
        return grad
