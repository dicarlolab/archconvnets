from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

layer_out = np.asarray(np.random.random((6,8)),dtype='single')

###########
t_start = time.time()
z3 = softmax_dlayer_in(layer_out)
t_cpu = time.time() - t_start

###

LAYER_OUT = [1, layer_out.shape]
OUT_BUFFER = [2, None]

nm.set_buffer(layer_out, LAYER_OUT[0])

#############
t_start = time.time()
nm.softmax_dlayer_in(LAYER_OUT, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

