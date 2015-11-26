from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

shift_out = np.asarray(np.random.random((16,3)),dtype='single')
w_interp = np.asarray(np.random.random((16,6)),dtype='single')

###########
t_start = time.time()
z3 = shift_w_dw_interp(shift_out)
t_cpu = time.time() - t_start

###
nm.set_buffer(shift_out,1)

#############
t_start = time.time()
nm.shift_w_dw_interp(1, w_interp.shape,2)
z3g = nm.return_buffer(2).reshape(z3.shape)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

