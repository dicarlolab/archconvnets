from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

a = np.asarray(np.random.random((16,3)),dtype='single')

s1 = 2.3
s2 = 3.1

A = nm.init_buffer(a)

###########
t_start = time.time()
a = a*s1 + s2
t_cpu = time.time() - t_start


#############
t_start = time.time()
nm.point_wise_add_scalar(A, scalar1=s1, scalar2=s2)
z3g = nm.return_buffer(A)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(a,z3g).sum()/np.single(np.prod(a.shape))

