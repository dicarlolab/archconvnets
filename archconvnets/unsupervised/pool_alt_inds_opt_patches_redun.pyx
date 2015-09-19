cimport numpy as npd
import numpy as np
import hashlib
import copy

''' conv_output: [n_imgs, n_filters, conv_sz, conv_sz]
	output_switches_x: [n_imgs, n_filters, output_sz, output_sz]
	imgs: [n_imgs, n_channels, img_sz, img_sz]
'''
def max_pool_locs_alt_patches_redun(npd.ndarray[npd.float32_t, ndim=4] conv_output, npd.ndarray[npd.int_t, ndim=4] output_switches_x, npd.ndarray[npd.int_t, ndim=4] output_switches_y, npd.ndarray[npd.float32_t, ndim=4] imgs, int s): 
	assert conv_output.shape[2] == conv_output.shape[3]
	assert conv_output.shape[0] == output_switches_x.shape[0]
	assert conv_output.shape[1] == output_switches_x.shape[1]
	assert output_switches_y.shape[0] == output_switches_x.shape[0]
	assert output_switches_y.shape[1] == output_switches_x.shape[1]
	assert output_switches_y.shape[2] == output_switches_x.shape[2]
	assert output_switches_y.shape[3] == output_switches_x.shape[3]
	
	cdef int conv_sz = conv_output.shape[1]
	cdef int x = 0
	cdef int y = 0
	cdef int filter
	cdef int n_filters = conv_output.shape[1]
	cdef int n_imgs = conv_output.shape[0]
	cdef int output_sz = output_switches_x.shape[2]
	cdef int a1_x_global
	cdef int a1_y_global
	cdef int n_channels = imgs.shape[1]
	
	cdef npd.ndarray[npd.int_t, ndim=1] n_patches = np.zeros((n_imgs), dtype='int') # unique patches per image
	cdef npd.ndarray[npd.float32_t, ndim=4] output = np.zeros((n_imgs, n_filters, output_sz, output_sz),dtype='single')
	cdef npd.ndarray[npd.int_t, ndim=4] pool_patch_inds = np.zeros((n_imgs, n_filters, output_sz, output_sz),dtype='int')
	cdef npd.ndarray[npd.float32_t, ndim=5] pool_patches = np.zeros((n_imgs, n_filters * output_sz * output_sz, n_channels, s, s),dtype='single')
	hash_x = np.empty((n_imgs, n_filters * output_sz * output_sz),dtype='int') 
	hash_y = np.empty((n_imgs, n_filters * output_sz * output_sz),dtype='int') 
	# ind = pool_patch_inds[img, filter, x, y] points to the corresponding patch -> pool_patches[img][ind]
	
	
	for img in range(n_imgs):
		for filter in range(n_filters):
			for x in range(output_sz):
				for y in range(output_sz):
					a1_x_global = output_switches_x[img, filter,x,y]
					a1_y_global = output_switches_y[img, filter,x,y]
					
					output[img, filter, x, y] = conv_output[img, filter, a1_x_global, a1_y_global]
					
					ind = np.nonzero((hash_x[img,:n_patches[img]] == a1_x_global) * (hash_y[img,:n_patches[img]] == a1_y_global))[0]
					
					if len(ind) == 0:
						pool_patches[img, n_patches[img]] = imgs[img, :, a1_x_global:a1_x_global+s, a1_y_global:a1_y_global+s]
						hash_x[img, n_patches[img]] = a1_x_global
						hash_y[img, n_patches[img]] = a1_y_global
						ind = n_patches[img]
						n_patches[img] += 1
					pool_patch_inds[img, filter, x, y] = ind
					
	
	return output, pool_patches, pool_patch_inds, n_patches

