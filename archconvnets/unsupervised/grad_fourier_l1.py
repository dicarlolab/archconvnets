import numpy as np
from scipy.stats.mstats import zscore
import time
import copy

def test_grad_fourier_l1(x, in_channels, filter_sz, n_filters, X, X2):
        sz2 = filter_sz**2
	x_in = copy.deepcopy(x)
        x_shape = x.shape
        x = np.float32(x.reshape((in_channels*(filter_sz**2), n_filters)))
        x = zscore(x,axis=0)
        x = x.reshape(x_shape)

        t_start = time.time()

        ################ fourier
        grad_f = np.zeros((in_channels, sz2, sz2, n_filters))
        Xx_sum = np.zeros(sz2)
        l = 0
        for channel in range(in_channels):
                for filter in range(n_filters):
                        x = x_in.reshape((in_channels, sz2, n_filters))[channel][:,filter]
                        Xx = np.dot(X,x)
                        Xx_sum += Xx
                        l += np.abs(Xx)
                        sign_mat = np.ones_like(Xx) - 2*(Xx < 0)
                        grad_f[channel][:,:,filter] = X * sign_mat[:,np.newaxis]
        grad_f = (grad_f).sum(1).ravel()
        fourier_loss = np.sum(np.abs( l))

        #########
	grad_f2 = np.zeros((in_channels, sz2, sz2, n_filters))
        Xx_sum = np.zeros(sz2)
        l = 0
        for channel in range(in_channels):
                for filter in range(n_filters):
                        x = x_in.reshape((in_channels, sz2, n_filters))[channel][:,filter]
                        Xx = np.dot(X2,x)
                        Xx_sum += Xx
                        l += np.abs(Xx)
                        sign_mat = np.ones_like(Xx) - 2*(Xx < 0)
                        grad_f2[channel][:,:,filter] = X2 * sign_mat[:,np.newaxis]
        #sign_mat2 = np.ones(sz2) - 2*(t2 > l)
        grad_f2 = (grad_f2).sum(1).ravel()
        fourier_loss += np.sum(np.abs( l))
        grad_f += grad_f2
	
	
        grad = grad_f
        loss = fourier_loss

        #print loss, fourier_loss, np.max(x_in)
        return np.double(loss), np.double(grad)

