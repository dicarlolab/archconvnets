from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

# O[IN] = 16,6
# O[IN_GATE] = 16,1
# O[CONTENT_SM] = 16,6
# o_prev = 16,6

interp_gate_out = np.asarray(np.random.random((16,1)),dtype='single')
o_content = np.asarray(np.random.random((16,6)),dtype='single')
o_prev = np.asarray(np.random.random((16,6)),dtype='single')

###########
t_start = time.time()
z3 = interpolate_do_prev(interp_gate_out, o_prev)
t_cpu = time.time() - t_start

###
nm.set_buffer(interp_gate_out,1)

#############
t_start = time.time()
nm.interpolate_do_prev(1, o_prev.shape,2)
z3g = nm.return_buffer(2).reshape(z3.shape)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

