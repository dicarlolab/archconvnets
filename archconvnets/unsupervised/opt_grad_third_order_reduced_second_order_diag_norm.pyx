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
        target_corr_mat = args[5]
        norm_second_order = args[6]
        N = args[7]
        f_start = time.time()
        cdef int in_dims = x.shape[0]/N
        cdef int n_eval = len(inds_i)
        cdef int n_triplets = math.factorial(in_dims)/(math.factorial(in_dims-3)*6)
        cdef Py_ssize_t i, m, n, r
        x = x.reshape((in_dims,N))
       
        cdef npd.ndarray[npd.float64_t, ndim=2] grad = np.zeros((in_dims, N))
        
        cdef npd.ndarray[npd.float64_t] x_mean = np.mean(x,axis=1)
        cdef npd.ndarray[npd.float64_t, ndim=2] x_no_mean = x - x_mean[:,np.newaxis]
        cdef npd.ndarray[npd.float64_t, ndim=2] x_no_mean_N = x_no_mean / N
        cdef npd.ndarray[npd.float64_t] x_std = np.std(x,axis=1)
        cdef npd.ndarray[npd.float64_t, ndim=2] x_no_mean_no_std = x_no_mean / (N*x_std[:,np.newaxis])

        cdef npd.ndarray[npd.float64_t, ndim=3] w = np.zeros((in_dims, in_dims, N))
        cdef npd.ndarray[npd.float64_t, ndim=2] w_std = np.zeros((in_dims, in_dims))
        for m in range(in_dims):
            for n in range(m,in_dims):
                w[n,m] = w[m,n] = x_no_mean[m]*x_no_mean[n]
                w_std[n,m] = w_std[m,n] = x_std[m]*x_std[n]
        cdef npd.ndarray[npd.float64_t, ndim=2] w_mean = np.mean(w, axis=2)
        cdef npd.ndarray[npd.float64_t, ndim=3] w_no_mean = w - w_mean[:,:,np.newaxis]
        
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
        numer /= sign_mat*(denom**2)
        denom = sign_mat / denom
        print loss
        # compute gradient for x[r,:]
        for r in range(in_dims):
            # find all terms in which x[r,:] contributes to:
            inds_global1 = np.nonzero(inds_i == r)[0]
            m1 = inds_m[inds_global1]
            n1 = inds_n[inds_global1]

            inds_global2 = np.nonzero(inds_m == r)[0]
            m2 = inds_i[inds_global2]
            n2 = inds_n[inds_global2]
            
            inds_global3 = np.nonzero(inds_n == r)[0]
            m3 = inds_m[inds_global3]
            n3 = inds_i[inds_global3]

            inds_global = np.concatenate((inds_global1, inds_global2, inds_global3))
            m_list = np.concatenate((m1, m2, m3))
            n_list = np.concatenate((n1, n2, n3))
            for ind_ind in range(len(inds_global)):
               ind = inds_global[ind_ind];
               m = m_list[ind_ind]; n = n_list[ind_ind]
               assert ((r == m) * (r == n)) == False, 'error: input indices cannot all be equal %i: %i: %i, %i, %i' % (ind_ind, ind, r,m,n)
               if (r == n): # double r
                    grad[r] += w_no_mean[r,m]*denom[ind] - numer[ind]*x_no_mean_N[r]*x_std[m]
               elif (r == m): # double r
                    grad[r] += w_no_mean[r,n]*denom[ind] - numer[ind]*x_no_mean_N[r]*x_std[n]
               else: # all unique inds (or double inds on the index we are not taking the gradient of)
                     grad[r] += w_no_mean[m,n]*denom[ind] - numer[ind]*w_std[m,n]*x_no_mean_no_std[r]
        thresh = 613677.907568*(50+2557)/2183.68977415
        if loss < thresh:#(1e8+2556.01800813):
                loss = -loss
                grad = -grad
        else:
                loss = -thresh#-1e8 - 2556.01800813
                grad = np.zeros_like(grad)
        ##########################################
        # 2nd order loss
        if norm_second_order != 0:
            corrs = (1-pdist(x,'correlation')) - target_corr_mat
            loss += norm_second_order*np.sum(np.abs(corrs))
            corr_mat = squareform(corrs)

            d_sum_n = np.sum(x_no_mean, axis=1) / n
            d2_sum_sqrt = np.sqrt(np.sum(x_no_mean**2, axis=1))
            d2_sum_sqrt2 = d2_sum_sqrt**2
            d_minus_sum_n = x_no_mean - d_sum_n[:,np.newaxis]
            d_minus_sum_n_div = d_minus_sum_n/d2_sum_sqrt[:,np.newaxis]
            d_dot_dT = np.dot(x_no_mean, x_no_mean.T)

            sign_mat2 = (np.ones((in_dims, in_dims)) - 2*(corr_mat < 0)) * norm_second_order

            for i in np.arange(in_dims):
                for j in np.arange(in_dims):
                    if i != j:
                         grad[i] += sign_mat2[i,j]*(d_minus_sum_n[j]*d2_sum_sqrt[i] - d_dot_dT[i,j]*d_minus_sum_n_div[i])/(d2_sum_sqrt[j]*d2_sum_sqrt2[i]) 
        
        if True: #(time.time() - time_save) >= 60:
            savemat(filename, {'x':x.reshape((np.prod(x.shape),1)), 'f': loss})
            time_save = time.time()
            print 'saving.........', loss, filename
        print '..elapsed time ', time.time() - f_start
        return loss, grad.reshape(in_dims*N)


