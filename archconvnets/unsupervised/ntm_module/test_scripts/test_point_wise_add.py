from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

a = np.asarray(np.random.random((16,3)),dtype='single')
b = np.asarray(np.random.random((16,3)),dtype='single')
s = 2.3

nm.set_buffer(a,1)
nm.set_buffer(b,2)

###########
t_start = time.time()
a += b*s
t_cpu = time.time() - t_start


#############
t_start = time.time()
nm.point_wise_add(1, 2, scalar=s)
z3g = nm.return_buffer(1).reshape(a.shape)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(a,z3g).sum()/np.single(np.prod(a.shape))

