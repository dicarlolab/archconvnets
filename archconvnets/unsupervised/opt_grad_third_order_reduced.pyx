import math
from scipy.spatial.distance import squareform
import copy
import time
from scipy.stats import pearsonr
import scipy.optimize
cimport numpy as npd
import numpy as np
import random
from scipy.spatial.distance import pdist
from scipy.io import loadmat
from scipy.io import savemat
from scipy.stats.mstats import zscore

global time_save
time_save = 0
def test_grad(x, *args): #npd.ndarray[npd.float64_t, ndim=1] x, *args): #x, *args):
        global time_save
        target = args[0]
        filename = args[1]
        inds_i = args[2]
        inds_m = args[3]
        inds_n = args[4]
        f_start = time.time()
        cdef int N = 64
        cdef int in_dims = x.shape[0]/N
        cdef int n_eval = len(inds_i)
        cdef int n_triplets = math.factorial(in_dims)/(math.factorial(in_dims-3)*6)
        cdef Py_ssize_t i, m, n, r
        x = x.reshape((in_dims,N))
       
        cdef npd.ndarray[npd.float64_t, ndim=2] grad = np.zeros((in_dims, N))
        
        cdef npd.ndarray[npd.float64_t] x_mean = np.mean(x,axis=1)
        cdef npd.ndarray[npd.float64_t, ndim=2] x_no_mean = x - x_mean[:,np.newaxis]
        cdef npd.ndarray[npd.float64_t] x_std = np.std(x,axis=1)
        cdef npd.ndarray[npd.float64_t, ndim=2] x_no_mean_no_std = x_no_mean / (N*x_std[:,np.newaxis])

        cdef npd.ndarray[npd.float64_t, ndim=3] w = np.zeros((in_dims, in_dims, N))
        cdef npd.ndarray[npd.float64_t, ndim=2] w_std = np.zeros((in_dims, in_dims))
        for m in range(in_dims):
            for n in range(m+1,in_dims):
                w[n,m] = w[m,n] = x_no_mean[m]*x_no_mean[n]
                w_std[n,m] = w_std[m,n] = x_std[m]*x_std[n]
        cdef npd.ndarray[npd.float64_t, ndim=2] w_mean = np.mean(w, axis=2)
        cdef npd.ndarray[npd.float64_t, ndim=3] w_no_mean = w - w_mean[:,:,np.newaxis]
        
        cdef npd.ndarray[npd.int32_t, ndim=3] ind_mat = -1*np.ones((in_dims, in_dims, in_dims), dtype='int32')
        ind_dic = {}
        cdef npd.ndarray[npd.float64_t] denom = np.zeros(n_eval)
        cdef npd.ndarray[npd.float64_t] numer = np.zeros(n_eval)
        cdef int ind = 0
        
        for ind in range(n_eval):
            i = inds_i[ind]; m = inds_m[ind]; n = inds_n[ind]
            
            numer[ind] = np.sum(x_no_mean[i]*w[m,n])
            denom[ind] = x_std[i]*w_std[m,n]
        
        cdef npd.ndarray[npd.float64_t] stat_mat = numer / denom
        loss = np.sum(np.abs(target - stat_mat))
        cdef npd.ndarray[npd.long_t] sign_mat =  1 - 2*(stat_mat < target)
        numer /= (denom**2)
        denom = 1 / denom
        print loss#, len(ind_dic.keys()), ind_dic.keys()[3]
        # compute gradient for x[r,m]
        for r in range(in_dims):
            for ind in np.nonzero(inds_i == r)[0]:
               m = inds_m[ind]; n = inds_n[ind]
               grad[r] += sign_mat[ind]*(w_no_mean[m,n]*denom[ind] - numer[ind]*w_std[m,n]*x_no_mean_no_std[r])
            for ind in np.nonzero(inds_m == r)[0]:
              m = inds_i[ind]; n = inds_n[ind]
              grad[r] += sign_mat[ind]*(w_no_mean[m,n]*denom[ind] - numer[ind]*w_std[m,n]*x_no_mean_no_std[r])
            for ind in np.nonzero(inds_n == r)[0]:
              m = inds_m[ind]; n = inds_i[ind]
              grad[r] += sign_mat[ind]*(w_no_mean[m,n]*denom[ind] - numer[ind]*w_std[m,n]*x_no_mean_no_std[r])
        if (time.time() - time_save) >= 60:
            savemat(filename, {'x':x.reshape((np.prod(x.shape),1)), 'f': loss})
            time_save = time.time()
            print 'saving.........', loss, filename
        print 'elapsed time ', time.time() - f_start
        return loss, grad.reshape(in_dims*N)

