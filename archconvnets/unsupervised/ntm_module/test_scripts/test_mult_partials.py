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

####
DG3UNDER_RELU_DG3UNDER = [1, dg3under_relu_dg3under.shape]
DG3UNDER_DW3UNDER = [2, dg3under_dw3under.shape]
OUT_BUFFER = [3, None]

nm.set_buffer(dg3under_relu_dg3under, DG3UNDER_RELU_DG3UNDER[0])
nm.set_buffer(dg3under_dw3under, DG3UNDER_DW3UNDER[0])

t_start = time.time()
nm.mult_partials(DG3UNDER_RELU_DG3UNDER, DG3UNDER_DW3UNDER, OUNDER[F_UNDER].ndim, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
