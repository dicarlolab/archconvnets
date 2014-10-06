from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from scipy.stats.mstats import zscore
import time
import copy

def test_grad_channel_corr(x, in_channels, filter_sz, n_filters):
        sz2 = filter_sz**2
	x_in = copy.deepcopy(x)
        x_shape = x.shape
        x = np.float32(x.reshape((in_channels*(filter_sz**2), n_filters)))

        t_start = time.time()
	n_channels_in = in_channels
	n_channels_find = n_filters

	grad_c = np.zeros((n_channels_in, filter_sz**2, n_channels_find))
        loss_c = 0
        for filter in range(n_channels_find):
                #print filter
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
        loss = -loss_c
        grad = -grad_c

        #print loss, fourier_loss, np.max(x_in)
        return np.double(loss.ravel()), np.double(grad.ravel())

