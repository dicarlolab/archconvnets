import math
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

filters_name = 'filters_layer1_4layermodel'
filename = 'opt_' + filters_name + '_thirdorder.mat'

time_save = time.time()

def test_grad(x):
        r = 14; q = 0
        x_in = copy.deepcopy(x)
        in_dims = n_channels_in*(filter_sz**2)
        x = copy.deepcopy(x_g)
        x[r,q] = x_in
      
        N = x.shape[1] 
        grad = np.zeros((in_dims, N))
        x_mean = np.mean(x,axis=1)
        x_no_mean = x - x_mean[:,np.newaxis] 
        corrs = (1-pdist(x,'correlation'))
	numer = np.sum((corrs - corrs.mean())*target)
        denom = np.std(corrs)*np.std(target)
	#return numer/denom#numer #denom
	return pearsonr(corrs,target)[0]

def test_grad_grad(x):
        r_r = 14; q = 0
        x_in = copy.deepcopy(x)
        in_dims = n_channels_in*(filter_sz**2)
        x = copy.deepcopy(x_g)
        x[r_r,q] = x_in

        N = x.shape[1]
        grad_s = np.zeros((in_dims, in_dims, N))
        x_mean = np.mean(x,axis=1)
        x_no_mean = x - x_mean[:,np.newaxis]
        corrs = (1-pdist(x,'correlation'))
        corr_mat = squareform(corrs); target_mat = squareform(target)
	numer = np.sum((corrs - corrs.mean())*target)
	denom = np.std(corrs)

        d_sum_n = np.mean(x_no_mean, axis=1)
        d2_sum_sqrt = np.sqrt(np.sum(x_no_mean**2, axis=1))
        d2_sum_sqrt2 = d2_sum_sqrt**2
        d_minus_sum_n = x_no_mean - d_sum_n[:,np.newaxis]
        d_minus_sum_n_div = d_minus_sum_n/d2_sum_sqrt[:,np.newaxis]
        d_dot_dT = np.dot(x_no_mean, x_no_mean.T)

        for i in np.arange(in_dims):
		for j in np.arange(in_dims):
	            if i != j:
                         grad_s[i,j] = (d_minus_sum_n[j]*d2_sum_sqrt[i] - d_dot_dT[i,j]*d_minus_sum_n_div[i])/(d2_sum_sqrt[j]*d2_sum_sqrt2[i])
	grad_s_mean = grad_s.sum(1)/len(corrs) # in_dims by  N
	grad_numer = np.zeros((in_dims, N))
	
	for r in range(in_dims):
		target_mat_temp = copy.deepcopy(target_mat)
		target_mat_temp[r] = 0
		target_mat_temp[:,r] = 0
		mean_term = squareform(target_mat_temp,checks=False)[:,np.newaxis]*grad_s_mean[r]
		grad_numer[r] = (np.sum((grad_s[r] - grad_s_mean[r])*target_mat[r][:,np.newaxis],axis=0) - mean_term.sum(0))

	grad_denom = np.sum((grad_s - grad_s_mean)*(corr_mat - corrs.mean())[:,:,np.newaxis],axis=1)/(denom*(N**2))
	grad = (grad_numer*denom - numer*grad_denom)/(denom**2)
	return grad[r_r,q]/(np.std(target)*N)

in_dims = n_channels_in*(filter_sz**2)
n_tuples = math.factorial(in_dims)/(math.factorial(in_dims-2)*2)
target = np.random.random(n_tuples)
x_g = np.random.random((in_dims, n_channels_find))
x0 = np.random.random((1,1)) #np.random.random((n_channels_in*filter_sz*filter_sz*n_channels_find,1))

print scipy.optimize.check_grad(test_grad, test_grad_grad, x0)

