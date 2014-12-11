cimport numpy as npd
import numpy as np

''' conv_output: 
'''
def max_pool_locs_alt(npd.ndarray[npd.float64_t, ndim=4] conv_output, npd.ndarray[npd.int_t, ndim=4] output_switches_x, npd.ndarray[npd.int_t, ndim=4] output_switches_y, int pool_stride=2, int pool_window_sz=3): 
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
	
	cdef npd.ndarray[npd.float64_t, ndim=4] output = np.zeros((n_filters, output_sz, output_sz, n_imgs))

	for filter in range(n_filters):
		for x in range(output_sz):
			for y in range(output_sz):
				for img in range(n_imgs):
					output[filter, x, y, img] = conv_output[filter, output_switches_x[filter, x, y, img], output_switches_y[filter, x, y, img], img]
	
	return output

