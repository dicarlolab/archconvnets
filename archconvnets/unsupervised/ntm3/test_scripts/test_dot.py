import numpy as np
import time
from ntm_core import *

N_IMGS = 2

f = np.single(np.random.random((2,3)))
x = np.single(np.random.random((N_IMGS,3,4)))

F = init_buffer(f)
X = init_buffer(x)

out = np.zeros((N_IMGS, f.shape[0], x.shape[-1]), dtype='single')

for i in range(N_IMGS):
	out[i] = np.dot(f,x[i])

LAYER_OUT = linear_F([F,X], additional_args=[False, False, True])

layer_out = return_buffer(LAYER_OUT)

print np.isclose(layer_out, out).sum()/np.single(np.prod(layer_out.shape))


