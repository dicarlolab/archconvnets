#from procs import *
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
neuron_ind = 3
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

n_channels_in = 3;
n_channels_out = 64
filter_sz = 5

n_channels_find = 64

filename = 'opt_cifar4.mat'

time_save = time.time()

def DFT_matrix_2d(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    A=np.multiply.outer(i.flatten(), i.flatten())
    B=np.multiply.outer(j.flatten(), j.flatten())
    omega = np.exp(-2*np.pi*1J/N)
    W = np.power(omega, A+B)/N
    return W

def test_grad(x):
        global n_channels_find
        global n_channels_in
        global n_channels_out
        global filter_sz
        global x_t
        global time_save
        x_in = copy.deepcopy(x)
        n = n_channels_find
        in_dims = n_channels_in*(filter_sz**2)
        x = np.reshape(x, (in_dims, n_channels_find))
        
        corrs = (1-pdist(x,'correlation')) - c_mat_input
        loss = np.sum(np.abs(corrs))
        corr_mat = squareform(corrs)
        
        grad = np.zeros((in_dims, n_channels_find))
        
        d = x - np.mean(x,axis=1)[:,np.newaxis]
        d_sum_n = np.sum(d, axis=1) / n
        d2_sum_sqrt = np.sqrt(np.sum(d**2, axis=1))
        d2_sum_sqrt2 = d2_sum_sqrt**2
        d_minus_sum_n = d - d_sum_n[:,np.newaxis]
        d_minus_sum_n_div = d_minus_sum_n/d2_sum_sqrt[:,np.newaxis]
        d_dot_dT = np.dot(d, d.T)
        
        sign_mat = np.ones((in_dims, in_dims)) - 2*(corr_mat < 0)

        for i in np.arange(in_dims):
            for j in np.arange(in_dims):
                if i != j:
                    grad[i] += sign_mat[i,j]*(d_minus_sum_n[j]*d2_sum_sqrt[i] - d_dot_dT[i,j]*d_minus_sum_n_div[i])/(d2_sum_sqrt[j]*d2_sum_sqrt2[i])             
        #print loss
        
	################
	sz2 = filter_sz**2
        x_in = copy.deepcopy(x)
        x_shape = x.shape
        x = np.float32(x.reshape((n_channels_in*(filter_sz**2), n_channels_find)))
        x = zscore(x,axis=0)
        x = x.reshape(x_shape)

        t_start = time.time()

        ################ fourier
        grad_f = np.zeros((n_channels_in, sz2, sz2, n_channels_find))
        Xx_sum = np.zeros(sz2)
        l = 0
        for channel in range(n_channels_in):
                for filter in range(n_channels_find):
                        x = x_in.reshape((n_channels_in, sz2, n_channels_find))[channel][:,filter]
                        Xx = np.dot(X,x)
                        Xx_sum += Xx
                        l += np.abs(Xx)
                        sign_mat = np.ones_like(Xx) - 2*(Xx < 0)
                        grad_f[channel][:,:,filter] = X * sign_mat[:,np.newaxis]
	#sign_mat2 = np.ones(sz2) - 2*(t > l)
        grad_f = (grad_f).sum(1).reshape((n_channels_in*sz2,n_channels_find))
        fourier_loss = np.sum(np.abs( l))
	
	grad_f2 = np.zeros((n_channels_in, sz2, sz2, n_channels_find))
	Xx_sum = np.zeros(sz2)
        l = 0
        for channel in range(n_channels_in):
                for filter in range(n_channels_find):
                        x = x_in.reshape((n_channels_in, sz2, n_channels_find))[channel][:,filter]
                        Xx = np.dot(X2,x)
                        Xx_sum += Xx
                        l += np.abs(Xx)
                        sign_mat = np.ones_like(Xx) - 2*(Xx < 0)
                        grad_f2[channel][:,:,filter] = X2 * sign_mat[:,np.newaxis]
        #sign_mat2 = np.ones(sz2) - 2*(t2 > l)
        grad_f2 = (grad_f2).sum(1).reshape((n_channels_in*sz2,n_channels_find))
        fourier_loss += np.sum(np.abs( l))
	grad_f += grad_f2
	
        #########
	scale_fourier = 1
	
	#if scale_fourier*fourier_loss < .3:
	#	print scale_fourier*fourier_loss
        #        fourier_loss = .3/scale_fourier
        #        grad_f = 0
	
        if (time.time() - time_save) >= 1:
            savemat(filename, {'x':x_in, 'f': loss})
            time_save = time.time()
            print 'saving.........', loss, fourier_loss*scale_fourier, filename
	
	grad += scale_fourier*grad_f
	loss += scale_fourier*fourier_loss
	
        return loss, grad.reshape(in_dims*n_channels_find)

X = np.real(DFT_matrix_2d(filter_sz))
X2 = np.imag(DFT_matrix_2d(filter_sz))
t = loadmat('/home/darren/fourier_target.mat')['t'].ravel()
t2 = loadmat('/home/darren/fourier_target2.mat')['t'].ravel()
model5 = unpickle('/home/darren/cifar_checkpoints/color/3layer/ConvNet__2014-08-11_19.47.17/120.5')
filters = copy.deepcopy(model5['model_state']['layers'][3]['inputLayers'][0]['weights'][0])
filters = filters.reshape((n_channels_in*(filter_sz**2), 64))#n_channels_out))

ta = copy.deepcopy(filters) #t = zscore(filters)
ta[np.isnan(ta)] = 0
c_mat_input = 1-pdist(ta,'correlation')

x0 = np.random.random((n_channels_in*filter_sz*filter_sz*n_channels_find,1))

t_start = time.time()
t_save = time.time()
x, f, d = scipy.optimize.fmin_l_bfgs_b(test_grad, x0)
print time.time() - t_start
savemat(filename, {'x': x, 'f':f,'d':d})

