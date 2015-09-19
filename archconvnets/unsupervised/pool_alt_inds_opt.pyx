cimport numpy as npd
import numpy as np

''' conv_output: 
'''
def max_pool_locs_alt(npd.ndarray[npd.float32_t, ndim=5] conv_output, npd.ndarray[npd.int_t, ndim=4] output_switches_x, npd.ndarray[npd.int_t, ndim=4] output_switches_y): 
	assert conv_output.shape[1] == conv_output.shape[2]
	assert conv_output.shape[0] == output_switches_x.shape[0]
	assert conv_output.shape[3] == output_switches_x.shape[3]
	assert output_switches_y.shape[0] == output_switches_x.shape[0]
	assert output_switches_y.shape[1] == output_switches_x.shape[1]
	assert output_switches_y.shape[2] == output_switches_x.shape[2]
	assert output_switches_y.shape[3] == output_switches_x.shape[3]
	
	cdef int conv_sz = conv_output.shape[1]
	cdef int x_loc = 0
	cdef int y_loc = 0
	cdef int x = 0
	cdef int y = 0
	cdef int filter
	cdef int n_filters = conv_output.shape[0]
	cdef int n_imgs = conv_output.shape[3]
	cdef int output_sz = output_switches_x.shape[1]
	cdef int set
	
	cdef npd.ndarray[npd.float32_t, ndim=5] output = np.zeros((n_filters, output_sz, output_sz, n_imgs, conv_output.shape[4]), dtype='single')

	for set in range(conv_output.shape[4]):
		for filter in range(n_filters):
			for x in range(output_sz):
				for y in range(output_sz):
					for img in range(n_imgs):
						output[filter, x, y, img, set] = conv_output[filter, output_switches_x[filter, x, y, img], output_switches_y[filter, x, y, img], img, set]
		
	return output

