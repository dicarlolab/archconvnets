from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

keys = np.asarray(np.random.random((16,6)),dtype='single')
beta_out = np.asarray(np.random.random((16,1)),dtype='single')

###########
t_start = time.time()
z3 = focus_key_dkeys(keys, beta_out)
t_cpu = time.time() - t_start

###
BETA_OUT = [1, beta_out.shape]
KEYS = [2, keys.shape]
OUT_BUFFER = [3, None]

nm.set_buffer(beta_out, BETA_OUT[0])

#############
t_start = time.time()
nm.focus_key_dkeys(BETA_OUT, KEYS, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

