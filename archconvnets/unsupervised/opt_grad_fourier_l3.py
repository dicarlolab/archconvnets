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

n_cpus = 8
n_channels_in = 256
n_channels_out = 256
filter_sz = 3
sz2 = filter_sz**2
weight_ind = 8#2#8
neuron_ind = 9#3#9

n_channels_find = n_channels_out

filename = 'opt_imgnet_l1.mat'

time_save = time.time()

def DFT_matrix_2d(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    A=np.multiply.outer(i.flatten(), i.flatten())
    B=np.multiply.outer(j.flatten(), j.flatten())
    omega = np.exp(-2*np.pi*1J/N)
    W = np.power(omega, A+B)/N
    return W

def second_order_grad(grad_shape, i_inds, in_dims, file_name):
	global neuron_ind, weight_ind
	x = loadmat(file_name)
	sign_mat = np.squeeze(x['sign_mat'])
	d_minus_sum_n = np.squeeze(x['d_minus_sum_n'])
	d2_sum_sqrt = np.squeeze(x['d2_sum_sqrt'])
	d_dot_dT = np.squeeze(x['d_dot_dT'])
	d_minus_sum_n_div = np.squeeze(x['d_minus_sum_n_div'])
	d2_sum_sqrt2 = np.squeeze(x['d2_sum_sqrt2'])
	grad = np.zeros(grad_shape,dtype='single')
	ind = 0
	for i in i_inds:
	    #if ind%250 == 0:
	    #	print ind, len(i_inds)
            for j in np.arange(in_dims):
                if i != j:
			if len(i_inds) == 1:
				grad[i] += sign_mat[i,j]*(d_minus_sum_n[j]*d2_sum_sqrt[i] - d_dot_dT[i,j]*d_minus_sum_n_div)/(d2_sum_sqrt[j]*d2_sum_sqrt2)
			else:
				grad[i] += sign_mat[i,j]*(d_minus_sum_n[j]*d2_sum_sqrt[i] - d_dot_dT[i,j]*d_minus_sum_n_div[ind])/(d2_sum_sqrt[j]*d2_sum_sqrt2[ind])
	    ind += 1
	savemat(file_name, {'grad': grad})
	return 0#grad

def test_grad(x):
        global n_channels_find
        global n_channels_in
        global n_channels_out
        global filter_sz
        global x_t
        global time_save
        t_start = time.time()
	x_in = copy.deepcopy(x)
        n = n_channels_find
        in_dims = n_channels_in*(filter_sz**2)
        x = np.reshape(x, (in_dims, n_channels_find))
       
	############################ channel corr
	grad_c = np.zeros((n_channels_in, filter_sz**2, n_channels_find))
        loss_c = 0
        for filter in range(n_channels_find):
                print filter
                x_t = copy.deepcopy(x[:, filter]).reshape((n_channels_in, filter_sz**2))
                corrs = 1-pdist(x_t,'correlation')
                loss_c += np.sum(np.abs(corrs))
                corr_mat = squareform(corrs)

                d = x_t - np.mean(x_t,axis=1)[:,np.newaxis]
                d_sum_n = np.sum(d, axis=1) / (filter_sz**2)
                d2_sum_sqrt = np.sqrt(np.sum(d**2, axis=1))
                d2_sum_sqrt2 = d2_sum_sqrt**2
                d_minus_sum_n = d - d_sum_n[:,np.newaxis]
                d_minus_sum_n_div = d_minus_sum_n/d2_sum_sqrt[:,np.newaxis]
                d_dot_dT = np.dot(d, d.T)

                sign_mat = np.ones((n_channels_in, n_channels_in)) - 2*(corr_mat < 0)

                for i in np.arange(n_channels_in):
                    for j in np.arange(n_channels_in):
                        if i != j:
                                grad_c[i,:,filter] += sign_mat[i,j]*(d_minus_sum_n[j]*d2_sum_sqrt[i] - d_dot_dT[i,j]*d_minus_sum_n_div[i])/(d2_sum_sqrt[j]*d2_sum_sqrt2[i])
	loss_c = -loss_c
	grad_c = -grad_c 
        ########################### second order
	print pearsonr((1-pdist(x,'correlation')), c_mat_input)
	corrs = (1-pdist(x,'correlation')) - c_mat_input
        loss = np.sum(np.abs(corrs))
        corr_mat = squareform(corrs)
        
        grad = np.zeros((in_dims, n_channels_find))
	grad_shape = grad.shape
        
        d = x - np.mean(x,axis=1)[:,np.newaxis]
        d_sum_n = np.sum(d, axis=1) / n
        d2_sum_sqrt = np.sqrt(np.sum(d**2, axis=1))
        d2_sum_sqrt2 = d2_sum_sqrt**2
        d_minus_sum_n = d - d_sum_n[:,np.newaxis]
        d_minus_sum_n_div = d_minus_sum_n/d2_sum_sqrt[:,np.newaxis]
        d_dot_dT = np.single(np.dot(d, d.T))
        
        sign_mat = np.ones((in_dims, in_dims)) - 2*(corr_mat < 0)
	sign_mat = sign_mat.astype('int8')
	
	d_minus_sum_n = np.single(d_minus_sum_n)
	d2_sum_sqrt = np.single(d2_sum_sqrt)
	d_minus_sum_n_div = np.single(d_minus_sum_n_div)
	d2_sum_sqrt2 = np.single(d2_sum_sqrt2)
	l = []
	n_per_cpu = np.int(in_dims / (n_cpus-1))
	if False:
		for cpu in range(n_cpus):
			remainder = 1
			if cpu == (n_cpus-1):
				remainder = in_dims % (n_cpus-1)
				i_inds = range(cpu*n_per_cpu, cpu*n_per_cpu + remainder)
			else:
				i_inds = range(cpu*n_per_cpu, (cpu+1)*n_per_cpu)
			if remainder != 0:
				file_name = '/tmp/test' + str(cpu) + '.mat'
				savemat(file_name, {'sign_mat':sign_mat, 'd_minus_sum_n':d_minus_sum_n, 'd2_sum_sqrt':d2_sum_sqrt, 'd_dot_dT':d_dot_dT, 
					'd_minus_sum_n_div':d_minus_sum_n_div[i_inds], 'd2_sum_sqrt2':d2_sum_sqrt2[i_inds]})
				l.append(proc(second_order_grad, grad_shape, i_inds, in_dims, file_name))
		results = call(l)
		results = tuple(results)
	else:
		for cpu in range(n_cpus):
                        remainder = 1
                        if cpu == (n_cpus-1):
                                remainder = in_dims % (n_cpus-1)
                                i_inds = range(cpu*n_per_cpu, cpu*n_per_cpu + remainder)
                        else:
                                i_inds = range(cpu*n_per_cpu, (cpu+1)*n_per_cpu)
                        if remainder != 0:
                                file_name = '/tmp/test' + str(cpu) + '.mat'
                                savemat(file_name, {'sign_mat':sign_mat, 'd_minus_sum_n':d_minus_sum_n, 'd2_sum_sqrt':d2_sum_sqrt, 'd_dot_dT':d_dot_dT,
                                        'd_minus_sum_n_div':d_minus_sum_n_div[i_inds], 'd2_sum_sqrt2':d2_sum_sqrt2[i_inds]})
                                second_order_grad(grad_shape, i_inds, in_dims, file_name)
	for cpu in range(n_cpus-1):
		file_name = '/tmp/test' + str(cpu) + '.mat'
		grad += loadmat(file_name)['grad']
	cpu += 1
	if (in_dims % (n_cpus-1)) != 0:
		file_name = '/tmp/test' + str(cpu) + '.mat'
		grad += loadmat(file_name)['grad']
        
	################
	sz2 = filter_sz**2
        x_in = copy.deepcopy(x)
        x_shape = x.shape
        x = np.float32(x.reshape((n_channels_in*(filter_sz**2), n_channels_find)))
        x = zscore(x,axis=0)
        x = x.reshape(x_shape)

        ################ fourier
	x_in = zscore(x_in)
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
	scale_channel = 1e-1###############1*1e-1#1e0#1e3
	scale_fourier = 5e-4############## 1*1e-4#1e-3#1
	
	thresh =0 #28066.5#33731.274635/2
	if fourier_loss < thresh:
		print '.....'
                fourier_loss = thresh
                grad_f = 0
	
        if (time.time() - time_save) >= 5:
            savemat(filename, {'x':x_in, 'f': loss})
            time_save = time.time()
	    s = model4['model_state']['layers'][weight_ind]['weights'][0].shape
	    model4['model_state']['layers'][neuron_ind]['inputLayers'][0]['weights'][0] = ((x_in).reshape(s)).astype('single')/100
	    model4['model_state']['layers'][weight_ind]['weights'][0] = ((x_in).reshape(s)).astype('single')/100
	    pickle('/home/darren/tmp_l3.model',model4)
            print 'saving.........', loss, fourier_loss*scale_fourier, scale_channel*loss_c, time.time() - t_start

	#grad = np.zeros_like(grad)
	#loss = np.zeros_like(loss)
	
	grad += scale_fourier*grad_f + scale_channel*grad_c.reshape(grad.shape)
	loss += scale_fourier*fourier_loss + scale_channel*loss_c
	
        return np.double(loss), np.double(grad.reshape(in_dims*n_channels_find))

X = np.real(DFT_matrix_2d(filter_sz))
X2 = np.imag(DFT_matrix_2d(filter_sz))

#model4 = unpickle('/home/darren/imgnet_3layer_256_linear_prelim.model')
#model4 = unpickle('/home/darren/imgnet_3layer_256_final.model')
#model4 = unpickle('/export/storage2/tmp2.model')
#model4 = unpickle('/home/darren/imgnet_3layer_48_linear.model')
#model4 = unpickle('/home/darren/tmp.model')
#model4 = unpickle('/home/darren/imgnet_3layer_48_optL1.model')
#model4 = unpickle('/home/darren/imgnet_3layer_48_optL3.model')
model4 = unpickle('/home/darren/tmp_l2.model')

#########################
os.system('python /home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py -f /home/darren/tmp_l2.model --write-features=pool2_6a --feature-path=/tmp/pool2 --train-range=0 --test-range=1238-1241')
###########

x_c = np.zeros((n_channels_in*3*3,0),dtype='float32')
for i in range(1238,1242):
    print i
    x = np.load('/tmp/pool2/data_batch_' + str(i))['data'].reshape((128,n_channels_in,15,15))
    x = x.transpose((1,2,3,0))
    x = x[:,6:6+3][:,:,6:6+3]
    x = x.reshape((n_channels_in*3*3,128))
    x_c = np.concatenate((x_c,x),axis=1)

t = zscore(x_c)
t[np.isnan(t)] = 0
c_mat_input = squareform(1-pdist(t,'correlation'))


#c_mat_input = np.squeeze(loadmat('/home/darren/pixels_zscore.mat')['f'])
#c_mat_input = squareform(np.squeeze(loadmat('/home/darren/pixels_zscore.mat')['f']))
for block in range(n_channels_in):
    for block2 in range(n_channels_in):
        #c_mat_input[block*49:(block+1)*49, block2*49:(block2+1)*49] -= np.mean(c_mat_input[block*49:(block+1)*49, block2*49:(block2+1)*49])
        if block != block2:
                c_mat_input[block*sz2:(block+1)*sz2, block2*sz2:(block2+1)*sz2] *= 2
                c_mat_input[block*sz2:(block+1)*sz2, block2*sz2:(block2+1)*sz2][c_mat_input[block*sz2:(block+1)*sz2, block2*sz2:(block2+1)*sz2] < 0] = 0
        c_mat_input[block*sz2:(block+1)*sz2, block2*sz2:(block2+1)*sz2] -= np.mean(c_mat_input[block*sz2:(block+1)*sz2, block2*sz2:(block2+1)*sz2])
c_mat_input = squareform(c_mat_input,checks=False)


#c_mat_input =  1-pdist(model4['model_state']['layers'][weight_ind]['weights'][0].reshape((n_channels_in*filter_sz*filter_sz, n_channels_out)),'correlation')

x0 = np.random.random((n_channels_in*filter_sz*filter_sz*n_channels_find,1))

t_start = time.time()
t_save = time.time()
x, f, d = scipy.optimize.fmin_l_bfgs_b(test_grad, x0)
print time.time() - t_start
savemat(filename, {'x': x, 'f':f,'d':d})

#python /home/darren/archconvnets_write/archconvnets_write/convnet/shownet.py -f /home/darren/tmp_l3.model --write-features=pool3_11a --feature-path=/tmp/hvm --train-range=0 --test-range=10025-10044

#python convnet.py -f /home/darren/tmp_l3.model --layer-params=/home/darren/archconvnets/archconvnets/convnet/ut_model_full/layer-params_no_bias_noL3.cfg --scale-rate=1 --save-path=/home/darren/tmp --data-path=/storage/imgnet128_img138_full --gpu=0
