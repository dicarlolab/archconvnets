cimport numpy as npd
import numpy as np

''' conv_output: 
'''
def max_pool_locs(npd.ndarray[npd.float64_t, ndim=4] conv_output, npd.ndarray[npd.float64_t, ndim=6] output_deriv, int pool_stride, int output_sz2, int pool_window_sz, int filter_sz, int in_channels): 
	assert conv_output.shape[1] == conv_output.shape[2]
	cdef int x_loc = 0
	cdef int y_loc = 0
	cdef int x
	cdef int y
	cdef int filter
	cdef int n_filters = conv_output.shape[0]
	cdef int n_imgs = conv_output.shape[3]

	cdef npd.ndarray[npd.float64_t, ndim=4] output = np.zeros((n_filters, output_sz2, output_sz2, n_imgs))#,dtype='float32')
	cdef npd.ndarray[npd.float64_t, ndim=7] output_deriv_max = np.zeros((in_channels, filter_sz, filter_sz, n_filters, output_sz2, output_sz2, n_imgs))#,dtype='float32')
	#cdef npd.ndarray[npd.float64_t] 
	cdef npd.ndarray[npd.float64_t, ndim=6] deriv_patch
	cdef npd.ndarray[npd.float64_t, ndim=5] deriv_patch_flat
	cdef npd.ndarray[npd.float64_t, ndim=3] output_patch
	cdef npd.ndarray[npd.float64_t, ndim=2] output_patch_flat

	for x in range(output_sz2):
		y_loc = 0
		for y in range(output_sz2):
			for filter in range(n_filters):
				output_patch = conv_output[filter,x_loc:x_loc+pool_window_sz, y_loc:y_loc+pool_window_sz]
				deriv_patch = output_deriv[:,:,:,x_loc:x_loc+pool_window_sz, y_loc:y_loc+pool_window_sz]

				output_patch_flat = output_patch.reshape((pool_window_sz**2, n_imgs))
				deriv_patch_flat = deriv_patch.reshape((in_channels, filter_sz, filter_sz, pool_window_sz**2, n_imgs))

				inds = np.argmax(output_patch_flat,axis=0)
				output[filter,x,y] = output_patch_flat[inds, range(n_imgs)]
				output_deriv_max[:,:,:,filter,x,y] = deriv_patch_flat[:,:,:,inds, range(n_imgs)]

			y_loc += pool_stride
		x_loc += pool_stride
	return output, output_deriv_max

