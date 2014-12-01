cimport numpy as npd
import numpy as np

''' filters: in_channels, filter_sz, filter_sz, n_filters
    imgs: in_channels, img_sz, img_sz, n_imgs
'''
def conv_block(npd.ndarray[npd.float64_t, ndim=4] filters, npd.ndarray[npd.float64_t, ndim=4] imgs, int stride=1): 
	assert filters.shape[1] == filters.shape[2]
	assert imgs.shape[1] == imgs.shape[2]
	assert imgs.shape[0] == filters.shape[0]
	cdef int x_loc = 0
	cdef int y_loc = 0
	cdef int x
	cdef int y
	cdef int n_filters = filters.shape[3]
	cdef int n_imgs = imgs.shape[3]
	cdef int img_sz = imgs.shape[1]
	cdef int filter_sz = filters.shape[1]
	cdef int in_channels = filters.shape[0]
	cdef output_sz = len(range(0, img_sz - filter_sz + 1, stride))
	cdef npd.ndarray[npd.float64_t, ndim=2] patch
	cdef npd.ndarray[npd.float64_t, ndim=4] output = np.zeros((n_filters, output_sz, output_sz, n_imgs))
	cdef npd.ndarray[npd.float64_t, ndim=2] filter_temp = filters.reshape((in_channels*filter_sz**2,n_filters)).T
	for x in range(output_sz):
		y_loc = 0
		for y in range(output_sz):
			patch = imgs[:,x_loc:x_loc+filter_sz, y_loc:y_loc+filter_sz].reshape((in_channels*filter_sz**2,n_imgs))
			output[:, x, y, :] = np.dot(filter_temp, patch)
			y_loc += stride
		x_loc += stride
	return output

