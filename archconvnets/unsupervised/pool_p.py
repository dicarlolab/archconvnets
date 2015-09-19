import numpy as np
import numexpr as ne
import time

''' imgs: in_channels, img_sz, img_sz, n_imgs
	conv output: n_filters, output_sz, output_sz, n_imgs
	filter_sz
	
	returns:
	pool_out: n_filters, pool_sz, pool_sz, n_imgs
	patch_out: in_channels, filter_sz, filter_sz, n_filters, pool_sz, pool_sz, n_imgs
	^ the patches that lead to each pooling output
'''
def pool_block(imgs, output, filter_sz):
	t_start = time.time()
	assert output.shape[1] == output.shape[2]
	assert imgs.shape[1] == imgs.shape[2]
	assert imgs.shape[3] == output.shape[3]
	n_filters = output.shape[0]
	n_imgs = imgs.shape[3]
	in_channels = imgs.shape[0]
	output_sz = output.shape[1]
	
	pool_sz = len(range(0,output_sz,2))
	pool_out = np.zeros((n_filters, pool_sz, pool_sz,n_imgs),dtype='single')
	patch_out = np.zeros((in_channels, filter_sz, filter_sz, n_filters, pool_sz, pool_sz, n_imgs),dtype='single')
	output_sz = output.shape[1]
	temp_patches = np.zeros((9,in_channels, filter_sz, filter_sz, n_imgs),dtype='single')
	x_ind = 0
	for x in range(0,output_sz-1,2):
		y_ind = 0
		for y in range(0,output_sz-1,2):
			block = output[:,x:x+3,y:y+3].reshape((n_filters, 9, n_imgs))
			offset_ind = 0
			for x_offset in range(3):
				for y_offset in range(3):
					temp_patches[offset_ind] = imgs[:,x+x_offset:x+x_offset+filter_sz,y+y_offset:y+y_offset+filter_sz]
					offset_ind += 1
			pool_out[:,x_ind,y_ind] = block.max(1)
			inds = block.argmax(1)
			for f in range(n_filters):
				patch_out[:,:,:,f,x_ind,y_ind] = temp_patches[inds[f],:,:,:,range(n_imgs)].transpose((1,2,3,0))
			y_ind += 1
		x_ind += 1
	print time.time() - t_start
	return pool_out, patch_out
			