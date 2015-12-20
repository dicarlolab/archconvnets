import archconvnets.unsupervised.ntm_module.ntm_module as nm
from archconvnets.unsupervised.ntm.ntm_gradients import *
import numpy as np
import time

# DWW = mult_partials_collapse__layers(derr_dor, DOR_DWW, OR[F]) # 38.5%
# print DWW[ADD].shape, derr_dor.shape, DOR_DWW[ADD].shape, OR[F].shape
# (16, 8, 9) (1, 16, 6) (16, 6, 16, 8, 9) (16, 6)

ADD = 0; F = 0
DOR_DWW = [None]; OR = [None]

derr_dor = np.asarray(np.random.random((1,16,6)),dtype='single')
DOR_DWW[ADD] = np.asarray(np.random.random((16, 6, 16, 8, 9)),dtype='single')
OR[F] = np.asarray(np.random.random((16, 6)),dtype='single')

t_start = time.time()
z3 = mult_partials_collapse(derr_dor, DOR_DWW[ADD], OR[F])
t_cpu = time.time() - t_start

'''nm.set_buffer(dg3under_relu_dg3under,1)
nm.set_buffer(dg3under_dw3under,2)

t_start = time.time()
nm.mult_partials(1,2, dg3under_relu_dg3under.shape, dg3under_dw3under.shape, OUNDER[F_UNDER].ndim, 3)
z3g = nm.return_buffer(3)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
'''