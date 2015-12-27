from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

gw = np.asarray(np.random.random((16, 6)),dtype='single')
add_out = np.asarray(np.random.random((16,8)),dtype='single')

###########
t_start = time.time()
z3 = add_mem_dgw(add_out)
t_cpu = time.time() - t_start

###
GW = nm.init_buffer(gw)
ADD_OUT = nm.init_buffer(add_out)
OUT_BUFFER = nm.init_buffer()

#############
t_start = time.time()
nm.add_mem_dgw(GW, ADD_OUT, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

