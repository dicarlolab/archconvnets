from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

f = np.asarray(np.random.random((3,13)),dtype='single')
x = np.asarray(np.random.random((13,12)),dtype='single')

###########
t_start = time.time()
z3 = linear_F_dF(f, x)
t_cpu = time.time() - t_start

###
F = [0, f.shape]
X = [1, x.shape]
OUT_BUFFER = [2, None]

nm.set_buffer(f, F[0])
nm.set_buffer(x, X[0])

#############
t_start = time.time()
nm.linear_F_dF(F, X, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

