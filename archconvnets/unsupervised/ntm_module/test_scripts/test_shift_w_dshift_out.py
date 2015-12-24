from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

shift_out = np.asarray(np.random.random((16,3)),dtype='single')
w_interp = np.asarray(np.random.random((16,6)),dtype='single')

###########
t_start = time.time()
z3 = shift_w_dshift_out(w_interp)
t_cpu = time.time() - t_start

###
W_INTERP = [1, w_interp.shape]
nm.set_buffer(w_interp,W_INTERP[0])

OUT_BUFFER = [2, None]

#############
t_start = time.time()
nm.shift_w_dshift_out(W_INTERP, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

