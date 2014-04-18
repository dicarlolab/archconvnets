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
from archconvnets.unsupervised.opt_grad_third_order_reduced_second_order import test_grad
import math

global filename
if True: #False:
	n_channels_in = 64
	filter_sz = 5

	n_channels_out = 64
	n_channels_find = 64

	filters_name = 'filters_layer2_4layermodel'
else:
	n_channels_in = 3
	filter_sz = 5
	
	n_channels_out = 64
	n_channels_find = 64
	
	filters_name = 'filters_layer1_4layermodel'

in_dims = n_channels_in*(filter_sz**2)
#################
N = n_channels_find
n_triplets = math.factorial(in_dims)/(math.factorial(in_dims-3)*6)
n_points = 65000#2775 #n_triplets #60000
filename = 'opt_' + filters_name + '_thirdorder_grad_npoints' + str(n_points) + '_nonrandom.mat'
inds_flat_eval = np.round(np.random.random(n_points) * n_triplets)
inds_flat_eval = inds_flat_eval[:n_points]
inds_flat_eval = np.sort(inds_flat_eval)
target = np.zeros(n_triplets) #n_points)
#inds_i = np.int32(np.round(np.random.random(n_points)*(in_dims-1))) #np.zeros(n_points,dtype='int32')
#inds_m = np.int32(np.round(np.random.random(n_points)*(in_dims-1))) #np.zeros(n_points,dtype='int32')
#inds_n = np.int32(np.round(np.random.random(n_points)*(in_dims-1))) #np.zeros(n_points,dtype='int32')
inds_i = np.zeros(n_triplets,dtype='int32')
inds_m = np.zeros(n_triplets,dtype='int32')
inds_n = np.zeros(n_triplets,dtype='int32')

filters = loadmat(filters_name + '.mat')['filters']
in_dims = n_channels_in*(filter_sz**2)
filters = filters.reshape((in_dims, n_channels_out))

x0 = np.random.random((in_dims, N))

#################
# compute target third order
x = copy.deepcopy(filters) #t = zscore(filters)
x[np.isnan(x)] = 0

x_no_mean = x - np.mean(x,axis=1)[:,np.newaxis]
x_std = np.std(x,axis=1)

w = np.zeros((in_dims, in_dims, N))
for m in np.arange(in_dims):
    for n in np.arange(in_dims):
	w[m,n] = x_no_mean[m]*x_no_mean[n]

numer = np.zeros(n_points)
denom = np.zeros(n_points)
ind = 0
for i in range(in_dims):
   for m in range(i+1, in_dims):
      for n in range(m+1, in_dims):
        numer = np.sum(x_no_mean[i]*w[m,n])
        denom = x_std[i]*x_std[m]*x_std[n]
        target[ind] = numer / denom
        inds_i[ind] = i; inds_m[ind] = m; inds_n[ind] = n
        ind += 1
reorder = np.argsort(-np.abs(target))
#inds = np.arange(len(reorder))
#random.shuffle(inds)
#reorder = reorder[inds]
target = target[reorder][:n_points]
inds_i = inds_i[reorder][:n_points]
inds_m = inds_m[reorder][:n_points]
inds_n = inds_n[reorder][:n_points]

#################
# compute target second order
x = copy.deepcopy(x0) #t = zscore(filters)
x[np.isnan(x)] = 0

x_no_mean = x - np.mean(x,axis=1)[:,np.newaxis]
x_std = np.std(x,axis=1)

w = np.zeros((in_dims, in_dims, N))
for m in np.arange(in_dims):
    for n in np.arange(in_dims):
        w[m,n] = x_no_mean[m]*x_no_mean[n]

numer = np.zeros(n_points)
denom = np.zeros(n_points)
for ind in range(n_points):
        #s = np.sort([inds_i[ind], inds_m[ind], inds_n[ind]])
        #if s[0] == s[1] or s[0] == s[2] or s[1] == s[2]:
        #  s[0] = 0; s[1] = 1; s[2] = np.round(np.random.random()*(in_dims-5))+3
        #inds_i[ind] = s[0]; inds_m[ind] = s[1]; inds_n[ind] = s[2]
        i = inds_i[ind]; m = inds_m[ind]; n = inds_n[ind]
        numer[ind] = np.sum(x_no_mean[i]*w[m,n])
        denom[ind] = x_std[i]*x_std[m]*x_std[n]
stat_mat = numer / denom

third_order_loss = np.sum(np.abs(target - stat_mat))

#########################
# second order

target_corr_mat = 1-pdist(filters,'correlation')
corrs = (1-pdist(x0,'correlation')) - target_corr_mat
second_order_loss = np.sum(np.abs(corrs))

norm_second_order = third_order_loss / second_order_loss
print norm_second_order

t_start = time.time()
time_save = time.time()
arg_list = (target, filename, inds_i, inds_m, inds_n, target_corr_mat, norm_second_order*0)
print filename
#print test_grad(x0, target, filename, inds_flat_eval)[0]#, target, filename)[0]
print 'starting...'
x, f, d = scipy.optimize.fmin_l_bfgs_b(test_grad, x0.reshape(in_dims*N,1), args=arg_list)
print f, time.time() - t_start

