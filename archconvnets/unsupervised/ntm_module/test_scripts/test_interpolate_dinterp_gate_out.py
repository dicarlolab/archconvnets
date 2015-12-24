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
z3 = interpolate_dinterp_gate_out(interp_gate_out, o_content, o_prev)
t_cpu = time.time() - t_start

###
O_CONTENT = [1, o_content.shape]
O_PREV = [2, o_prev.shape]
OUT_BUFFER = [3, None]

nm.set_buffer(o_content, O_CONTENT[0])
nm.set_buffer(o_prev, O_PREV[0])

#############
t_start = time.time()
nm.interpolate_dinterp_gate_out(O_CONTENT, O_PREV, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

