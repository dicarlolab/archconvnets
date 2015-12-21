import archconvnets.unsupervised.ntm_module.ntm_module as nm
from archconvnets.unsupervised.ntm.ntm_gradients import *
import numpy as np
import time

F_UNDER = 0
OUNDER = [None]

dg3under_relu_dg3under = np.asarray(np.random.random((9,9)),dtype='single')
dg3under_dw3under = np.asarray(np.random.random((9,9,22)),dtype='single')
OUNDER[F_UNDER] = np.asarray(np.random.random((9)),dtype='single')

t_start = time.time()
z3 = mult_partials(dg3under_relu_dg3under, dg3under_dw3under, OUNDER[F_UNDER])
t_cpu = time.time() - t_start

nm.set_buffer(dg3under_relu_dg3under,1)
nm.set_buffer(dg3under_dw3under,2)

t_start = time.time()
nm.mult_partials(1, dg3under_relu_dg3under.shape, 2, dg3under_dw3under.shape, OUNDER[F_UNDER].ndim, 3)
z3g = nm.return_buffer(3)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
