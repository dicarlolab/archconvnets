from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

a = np.asarray(np.random.random((16,3)),dtype='single')
b = np.asarray(np.random.random((16,3)),dtype='single')
s = 2.3

A = [0, a.shape]
B = [1, b.shape]

nm.set_buffer(a, A[0])
nm.set_buffer(b, B[0])

###########
t_start = time.time()
a += b*s
t_cpu = time.time() - t_start


#############
t_start = time.time()
nm.point_wise_add(A, B, scalar=s)
z3g = nm.return_buffer(A)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(a,z3g).sum()/np.single(np.prod(a.shape))

