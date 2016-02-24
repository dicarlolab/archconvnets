import numpy as np
import time
from archconvnets.unsupervised.ntm3.ntm_core import *

dim_above = 4
n_imgs = 2
n_filters = 7
img_sz = 4
out_sz = img_sz / 2
pool_width = 3
stride = 2
dim_above = 5

imgs = np.single(np.random.random((n_imgs, n_filters, img_sz,img_sz))) - .5
deriv_above = np.single(np.random.random((n_imgs, dim_above, n_filters, out_sz,out_sz))) - .5

IMGS = init_buffer(imgs)
DERIV_ABOVE = init_buffer(deriv_above)
LAYER_OUT = max_pool([IMGS])

out = return_buffer(LAYER_OUT)
dm = return_buffer(max_pool_dinput([IMGS], LAYER_OUT, DERIV_ABOVE))

out_cpu = np.zeros((n_imgs, n_filters, out_sz, out_sz), dtype='single')
dm_cpu = np.zeros((n_imgs, dim_above, n_filters, img_sz, img_sz), dtype='single')

for img in range(n_imgs):
	for filter in range(n_filters):
		for x in range(out_sz):
			for y in range(out_sz):
				window = imgs[img, filter, x*stride:x*stride+pool_width, y*stride:y*stride+pool_width] #.ravel()
				y_width = np.min([img_sz - y*stride, pool_width])
				
				window = window.ravel()
				out_cpu[img, filter, x,y] = np.max(window)
				
				max_loc = np.argmax(window)
				
				x_loc = max_loc / y_width
				y_loc = max_loc % y_width
				
				dm_cpu[img, :, filter, x*stride + x_loc, y*stride + y_loc] = deriv_above[img, :, filter, x,y]
				
print np.isclose(out_cpu,out).sum()/np.single(np.prod(out.shape))
print np.isclose(dm_cpu,dm).sum()/np.single(np.prod(dm.shape))
