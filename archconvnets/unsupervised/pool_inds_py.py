import numpy as np

# conv_output: imgs, filters, sz, sz
def max_pool_locs(conv_output, pool_stride=2, pool_window_sz=3, PAD=0): 
	assert conv_output.shape[2] == conv_output.shape[3]
	
	conv_sz = conv_output.shape[2]
	n_filters = conv_output.shape[1]
	n_imgs = conv_output.shape[0]
	output_sz = len(range(0,conv_sz-pool_window_sz,pool_stride))
	
	output = np.zeros((n_imgs, n_filters, output_sz, output_sz), dtype='single')
	output_switches_x = np.zeros((n_imgs, n_filters, output_sz, output_sz),dtype='int')
	output_switches_y = np.zeros((n_imgs, n_filters, output_sz, output_sz),dtype='int')
	
	x = 0
	for x_loc in range(0,conv_sz-pool_window_sz,pool_stride):
		y = 0
		for y_loc in range(0,conv_sz-pool_window_sz,pool_stride):
			for filter in range(n_filters):
				for img in range(n_imgs):
					output_patch = conv_output[img,filter,x_loc:x_loc+pool_window_sz, y_loc:y_loc+pool_window_sz]
					output_patch_flat = output_patch.ravel()

					ind = np.argmax(output_patch_flat)
					inds = np.unravel_index(ind, output_patch.shape)
					global_inds = inds + np.array([x_loc, y_loc])
					
					if global_inds[0] < PAD:
						global_inds[0] = PAD
					if global_inds[1] < PAD:
						global_inds[1] = PAD
					if conv_sz - PAD <= global_inds[0]:
						global_inds[0] = conv_sz - PAD - 1
					if conv_sz - PAD <= global_inds[1]:
						global_inds[1] = conv_sz - PAD - 1

					output[img,filter,x,y] = conv_output[img,filter,global_inds[0],global_inds[1]]
					output_switches_x[img,filter,x,y] = global_inds[0]
					output_switches_y[img,filter,x,y] = global_inds[1]

			y += 1
		x += 1
		
	return output, output_switches_x, output_switches_y

