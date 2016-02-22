import numpy as np
import time
from archconvnets.unsupervised.ntm3.ntm_core import *

PAD = 2

dim_above = 4
n_imgs = 2
n_filters = 7
filter_sz = 5

out_sz = 32 + PAD*2 - filter_sz + 1

imgs = np.single(np.random.random((n_imgs, 3,32,32))) - .5
filters = np.single(np.random.random((n_filters, 3, filter_sz,filter_sz)))
deriv_above = np.single(np.random.random((n_imgs, dim_above, n_filters, out_sz, out_sz)))

#imgs = np.ones_like(imgs)
#filters = np.ones_like(filters)
#deriv_above = np.ones_like(deriv_above)

IMGS = init_buffer(imgs)
FILTERS = init_buffer(filters)
DERIV_ABOVE = init_buffer(deriv_above)

out = return_buffer(conv((FILTERS, IMGS), additional_args=[PAD]))

####
imgs_pad = np.zeros(imgs.shape[:2] + (32 + 2*PAD,)*2, dtype='single')
imgs_pad[:,:, PAD:PAD+32, PAD:PAD+32] = imgs

out_cpu = np.zeros((n_imgs, n_filters, out_sz, out_sz), dtype='single')

filters_temp = filters.reshape((n_filters, 3*filter_sz*filter_sz)).T

for loc_x in range(out_sz):
	for loc_y in range(out_sz):
		imgs_temp = imgs_pad[:,:,loc_x:loc_x+filter_sz, loc_y:loc_y+filter_sz].reshape((n_imgs, 3*filter_sz*filter_sz))
		out_cpu[:,:, loc_x, loc_y] = np.dot(imgs_temp, filters_temp)
		
		
# deriv_above [n_imgs, dim_above, n_filters, out_sz, out_sz]
# F [n_filters, 3, f_sz, f_sz]
# imgs [n_imgs, 3, in_sz, in_sz]

# dF = [n_imgs, n_filters, out_sz, out_sz, n_filters, 3, f_sz, f_sz]

# ... dF = [n_imgs, out_sz, out_sz, 3, f_sz, f_sz]

# deriv_above * dF = [n_imgs, dim_above, n_filters, 3, f_sz, f_sz]

dF = np.zeros((n_imgs, out_sz, out_sz, 3, filter_sz, filter_sz), dtype='single')

for loc_x in range(out_sz):
	for loc_y in range(out_sz):
		for f1 in range(filter_sz):
			for f2 in range(filter_sz):
				dF[:, loc_x, loc_y, :, f1, f2] = imgs_pad[:,:, loc_x + f1, loc_y + f2]

dF_cpu = np.einsum(deriv_above, range(5), dF, [0, 3,4, 5,6,7], [0,1,2, 5,6,7])

dF_gpu = return_buffer(conv_dfilter((FILTERS, IMGS), init_buffer(out), DERIV_ABOVE, additional_args=[PAD]))

print np.isclose(dF_gpu, dF_cpu).sum() / np.single(np.prod(dF_gpu.shape))