from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.stats.mstats import zscore
import time
import numpy as np
import copy

def test_grad_transpose(x, in_channels, filter_sz, n_filters):
        x_in = copy.deepcopy(x)
        x_shape = x.shape
        x = np.float32(x.reshape((in_channels*(filter_sz**2), n_filters)))
        x = zscore(x,axis=0)
        x = x.reshape(x_shape)

        t_start = time.time()

        ########## transpose
        x = np.reshape(x, (in_channels*(filter_sz**2), n_filters)).T

        corrs = (1-pdist(x,'correlation'))
        loss_t = np.sum(np.abs(corrs))
        corr_mat = squareform(corrs)

        grad_t = np.zeros((in_channels*(filter_sz**2), n_filters),dtype='float32').T

        d = x - np.mean(x,axis=1)[:,np.newaxis]
        d_sum_n = np.sum(d, axis=1) / (in_channels*(filter_sz**2))
        d2_sum_sqrt = np.sqrt(np.sum(d**2, axis=1))
        d2_sum_sqrt2 = d2_sum_sqrt**2
        d_minus_sum_n = d - d_sum_n[:,np.newaxis]
        d_minus_sum_n_div = d_minus_sum_n/d2_sum_sqrt[:,np.newaxis]
        d_dot_dT = np.dot(d, d.T)

        sign_mat = np.ones((n_filters, n_filters)) - 2*(corr_mat < 0)

        for i in np.arange(n_filters):
                for j in np.arange(n_filters):
                        if i != j:
                                grad_t[i] += sign_mat[i,j]*(d_minus_sum_n[j]*d2_sum_sqrt[i] - d_dot_dT[i,j]*d_minus_sum_n_div[i])/(d2_sum_sqrt[j]*d2_sum_sqrt2[i])
        grad_t = grad_t.T.ravel()

        #if np.mean(np.abs(corrs)) < 0.15:
        #        loss_t = 0
        #        grad_t = 0

        loss = loss_t
        grad = grad_t

        #print loss, time.time() - t_start, np.mean(np.abs(corrs)), np.max(x_in)
        return np.double(loss), np.double(grad)

