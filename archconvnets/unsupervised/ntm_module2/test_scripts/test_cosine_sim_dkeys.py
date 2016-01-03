from archconvnets.unsupervised.ntm2.ntm_gradients import *
import archconvnets.unsupervised.ntm_module2.ntm_module2 as nm
import numpy as np
import time

keys = np.asarray(np.random.random((12,8)),dtype='single')
mem = np.asarray(np.random.random((6,8)),dtype='single')

############
t_start = time.time()
z3 = cosine_sim_dkeys((keys, mem))
t_cpu = time.time() - t_start

####
KEYS = nm.init_buffer(keys)
MEM = nm.init_buffer(mem)


##############
t_start = time.time()
OUT_BUFFER = nm.cosine_sim_dkeys((KEYS, MEM))
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))