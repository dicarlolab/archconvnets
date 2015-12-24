from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

w = np.asarray(np.random.random((16,6)),dtype='single')
gamma = np.asarray(np.random.random((16,1)),dtype='single')

###########
t_start = time.time()
z3 = dsharpen_dw(w, gamma)
t_cpu = time.time() - t_start

###
W = [1, w.shape]
GAMMA = [2, gamma.shape]
OUT_BUFFER = [3, None]

nm.set_buffer(w, W[0])
nm.set_buffer(gamma, GAMMA[0])

#############
t_start = time.time()
nm.sharpen_dw(W, GAMMA, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

