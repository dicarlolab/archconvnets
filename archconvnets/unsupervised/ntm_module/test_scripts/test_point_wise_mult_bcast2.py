from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

a = np.asarray(np.random.random((16,3,5,6)),dtype='single')
b = np.asarray(np.random.random((16,3)),dtype='single')
s = 2.3

A = nm.init_buffer(a)
B = nm.init_buffer(b)

###########
t_start = time.time()
a *= b[:,:,np.newaxis,np.newaxis]*s
t_cpu = time.time() - t_start


#############
t_start = time.time()
nm.point_wise_mult_bcast2(A, B, scalar=s)
z3g = nm.return_buffer(A)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(a,z3g).sum()/np.single(np.prod(a.shape))

