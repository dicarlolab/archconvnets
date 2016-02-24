import numpy as np
import time
from archconvnets.unsupervised.ntm3.ntm_core import *

n_imgs = 64
n_filters = 32
img_sz = 32
out_sz = img_sz / 2
pool_width = 3
stride = 2
dim_above = 400

imgs = np.single(np.random.random((n_imgs, n_filters, img_sz,img_sz))) - .5
deriv_above = np.single(np.random.random((n_imgs, dim_above, n_filters, out_sz,out_sz))) - .5

IMGS = init_buffer(imgs)
DERIV_ABOVE = init_buffer(deriv_above)
LAYER_OUT = max_pool([IMGS])

out = return_buffer(LAYER_OUT)

t_gpu = time.time()
DM = max_pool_dinput([IMGS], LAYER_OUT, DERIV_ABOVE)
print time.time() - t_gpu
