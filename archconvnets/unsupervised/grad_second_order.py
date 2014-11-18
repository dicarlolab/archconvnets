import copy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np

def test_grad_second_order(x, in_channels, filter_sz, n_filters, c_mat_input):
	n_channels_find = n_filters
        x_in = copy.deepcopy(x)
        n = n_channels_find
        in_dims = in_channels*(filter_sz**2)
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

        return loss, grad.reshape(in_dims*n_channels_find)
