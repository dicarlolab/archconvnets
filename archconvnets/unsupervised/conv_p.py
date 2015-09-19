import numpy as np
import numexpr as ne
import time

''' filters: in_channels, filter_sz, filter_sz, n_filters
    imgs: in_channels, img_sz, img_sz, n_imgs
	output: n_filters, output_sz, output_sz, n_imgs
'''
def conv_block(filters, imgs, stride=1, max_el=3360688123): 
	t_start = time.time()
	filters = np.single(filters); imgs = np.single(imgs)
	assert filters.shape[1] == filters.shape[2]
	assert imgs.shape[1] == imgs.shape[2]
	assert imgs.shape[0] == filters.shape[0]
	x_loc = 0
	y_loc = 0
	n_filters = filters.shape[3]
	n_imgs = imgs.shape[3]
	img_sz = imgs.shape[1]
	filter_sz = filters.shape[1]
	in_channels = filters.shape[0]
	output_sz = len(range(0, img_sz - filter_sz + 1, stride))
	filter_temp = filters.reshape((in_channels*filter_sz**2,n_filters)).T.reshape((n_filters,in_channels*filter_sz*filter_sz,1,1,1))	
	
	patches = np.zeros((1, in_channels*filter_sz*filter_sz,output_sz, output_sz, n_imgs),dtype='single')
	for x in range(output_sz):
		y_loc = 0
		for y in range(output_sz):
			patches[0,:,x,y] = imgs[:,x_loc:x_loc+filter_sz, y_loc:y_loc+filter_sz].reshape((1,in_channels*filter_sz*filter_sz,n_imgs))
			y_loc += stride
		x_loc += stride
	
	total_el = (n_filters*in_channels*filter_sz*filter_sz*output_sz*output_sz*n_imgs)
	n_groups = np.int(np.ceil(np.single(total_el) / max_el))
	imgs_per_group = np.int(np.floor(128.0/n_groups))
	#print n_groups
	#print imgs_per_group
	
	output = np.zeros((n_filters,output_sz,output_sz,n_imgs),dtype='single')
	for group in range(n_groups):
		p_t = patches[:,:,:,:,group*imgs_per_group:(group+1)*imgs_per_group]
		output[:,:,:,group*imgs_per_group:(group+1)*imgs_per_group] = ne.evaluate('filter_temp*p_t').sum(1)
	print time.time() - t_start
	return output

