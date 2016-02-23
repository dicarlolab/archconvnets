import numpy as np
import time
from archconvnets.unsupervised.ntm3.ntm_core import *

free_all_buffers()

PAD = 2

dim_above = 6*8*5
n_imgs = 32
n_filters = 48
filter_sz = 5

out_sz = 32 + PAD*2 - filter_sz + 1

imgs = np.single(np.random.random((n_imgs, 3,32,32))) - .5
filters = np.single(np.random.random((n_filters, 3, filter_sz,filter_sz)))
deriv_above = np.single(np.random.random((n_imgs, dim_above, n_filters, out_sz, out_sz)))

IMGS = init_buffer(imgs)
FILTERS = init_buffer(filters)
DERIV_ABOVE = init_buffer(deriv_above)

out = return_buffer(conv((FILTERS, IMGS), additional_args=[PAD]))

t_start = time.time()
dF_gpu = return_buffer(conv_dfilter((FILTERS, IMGS), init_buffer(out), DERIV_ABOVE, additional_args=[PAD]))

print time.time() - t_start
