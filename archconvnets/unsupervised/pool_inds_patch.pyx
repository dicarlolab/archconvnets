cimport numpy as npd
import numpy as np

''' conv_output: 
'''
def max_pool_locs_patches(npd.ndarray[npd.float64_t, ndim=4] conv_output, npd.ndarray[npd.float64_t, ndim=4] imgs, int s, int pool_stride=2, int pool_window_sz=3): 
	assert conv_output.shape[1] == conv_output.shape[2]
	assert imgs.shape[1] == imgs.shape[2]
	assert imgs.shape[3] == conv_output.shape[3]
	
	cdef int conv_sz = conv_output.shape[1]
	cdef int x_loc = 0
	cdef int y_loc = 0
	cdef int x = 0
	cdef int y = 0
	cdef int filter
	cdef int n_filters = conv_output.shape[0]
	cdef int n_imgs = conv_output.shape[3]
	cdef int n_channels = imgs.shape[0]
	cdef int img
	cdef int a1_x_global
	cdef int a1_y_global
	cdef int output_sz = len(range(0,conv_sz-pool_window_sz,pool_stride))
	
	cdef npd.ndarray[npd.float64_t, ndim=4] output = np.zeros((n_filters, output_sz, output_sz, n_imgs))
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches_x = np.zeros((n_filters, output_sz, output_sz, n_imgs),dtype='int')
	cdef npd.ndarray[npd.int_t, ndim=4] output_switches_y = np.zeros((n_filters, output_sz, output_sz, n_imgs),dtype='int')
	cdef npd.ndarray[npd.float64_t, ndim=3] output_patch
	cdef npd.ndarray[npd.float64_t, ndim=2] output_patch_flat
	cdef npd.ndarray[npd.int_t, ndim=1] inds
	cdef npd.ndarray[npd.float64_t, ndim=7] pool_patches = np.zeros((n_filters, output_sz, output_sz, n_imgs, n_channels, s, s))

	for x_loc in range(0,conv_sz-pool_window_sz,pool_stride):
		y = 0
		for y_loc in range(0,conv_sz-pool_window_sz,pool_stride):
			for filter in range(n_filters):
				output_patch = conv_output[filter,x_loc:x_loc+pool_window_sz, y_loc:y_loc+pool_window_sz]
				output_patch_flat = output_patch.reshape((output_patch.shape[0]*output_patch.shape[1], n_imgs))

				inds = np.argmax(output_patch_flat,axis=0)
				global_inds = np.asarray(np.unravel_index(inds, (output_patch.shape[0],output_patch.shape[1]))) + np.array([x_loc, y_loc])[:,np.newaxis]
				output[filter,x,y] = output_patch_flat[inds, range(n_imgs)]
				output_switches_x[filter,x,y] = global_inds[0]
				output_switches_y[filter,x,y] = global_inds[1]
				
				for img in range(n_imgs):
					a1_x_global = output_switches_x[filter,x,y,img]
					a1_y_global = output_switches_y[filter,x,y,img]
					
					pool_patches[filter, x, y, img] = imgs[:, a1_x_global:a1_x_global+s, a1_y_global:a1_y_global+s, img]
				#pool_patches[filter, x, y, img, channel, xf, yf] = imgs[channel, a1_x_global + xf, a1_y_global + yf, img]
			y += 1
		x += 1
	return output, output_switches_x, output_switches_y, pool_patches


	

'''cimport numpy as npd
import numpy as np

# select patches (imgs) based on pool locations (output_switches_)

def max_pool_patches(npd.ndarray[npd.float64_t, ndim=4] imgs, npd.ndarray[npd.int_t, ndim=4] output_switches_x, npd.ndarray[npd.int_t, ndim=4] output_switches_y, int s): 
	assert imgs.shape[1] == imgs.shape[2]
	assert imgs.shape[3] == output_switches_x.shape[3]
	assert output_switches_y.shape[0] == output_switches_x.shape[0]
	assert output_switches_y.shape[1] == output_switches_x.shape[1]
	assert output_switches_y.shape[2] == output_switches_x.shape[2]
	assert output_switches_y.shape[3] == output_switches_x.shape[3]
	
	cdef int n_channels = imgs.shape[0]
	cdef int pool_sz = output_switches_x.shape[1]
	cdef int xf = 0
	cdef int yf = 0
	cdef int x = 0
	cdef int y = 0
	cdef int filter
	cdef int n_filters = output_switches_x.shape[0]
	cdef int n_imgs = imgs.shape[3]
	cdef int output_sz = output_switches_x.shape[1]
	cdef int channel
	cdef int img
	cdef npd.ndarray[npd.float64_t, ndim=7] pool_patches = np.zeros((n_filters, output_sz, output_sz, n_imgs, n_channels, s, s))

	for filter in range(n_filters):
		for x in range(pool_sz):
			for y in range(pool_sz):
				for img in range(n_imgs):
					a1_x_global = output_switches_x[filter, x, y, img]
					a1_y_global = output_switches_y[filter, x, y, img]
					for channel in range(n_channels):
						for xf in range(s):
							for yf in range(s):
								pool_patches[filter, x, y, img, channel, xf, yf] = imgs[channel, a1_x_global + xf, a1_y_global + yf, img]
	
	return pool_patches

'''