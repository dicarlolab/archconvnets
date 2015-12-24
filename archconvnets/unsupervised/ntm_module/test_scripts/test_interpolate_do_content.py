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
z3 = interpolate_do_content(interp_gate_out, o_content)
t_cpu = time.time() - t_start

###
INTERP_GATE_OUT = [1, interp_gate_out.shape]
O_PREV = [2, o_prev.shape]
OUT_BUFFER = [3, None]

nm.set_buffer(interp_gate_out, INTERP_GATE_OUT[0])

#############
t_start = time.time()
nm.interpolate_do_content(INTERP_GATE_OUT, O_PREV, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

