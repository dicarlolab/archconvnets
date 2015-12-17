from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

F = np.asarray(np.random.random((3,13)),dtype='single')
x = np.asarray(np.random.random((13,12)),dtype='single')

###########
t_start = time.time()
z3 = linear_F_dF(F, x)
t_cpu = time.time() - t_start

###
nm.set_buffer(x,1)

#############
t_start = time.time()
nm.linear_F_dF(1,x.shape, F.shape,2)
z3g = nm.return_buffer(2).reshape(z3.shape)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

