from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

w = np.asarray(np.random.random((16,6)),dtype='single')
gamma = np.asarray(np.random.random((16,1)),dtype='single')

###########
t_start = time.time()
z3 = dsharpen_dw(w, gamma)
t_cpu = time.time() - t_start

#######
t_start = time.time()
z34 = nm.dsharpen_dw_cpu(w, gamma)
t_cpu2 = time.time() - t_start

###
nm.set_buffer(w,1)
nm.set_buffer(gamma,2)

#############
t_start = time.time()
nm.dsharpen_dw(1,w.shape, 2, gamma.shape, 3)
z3g = nm.return_buffer(3).reshape(z3.shape)
t_gpu = time.time() - t_start

print t_cpu, t_cpu2, t_cpu/t_cpu2, np.isclose(z3,z34).sum()/np.single(np.prod(z3.shape))
print t_cpu2, t_gpu, t_cpu2/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

