from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

layer_out = np.asarray(np.random.random((6,8)),dtype='single')

t_start = time.time()
z = softmax_dlayer_in_nsum(layer_out)
t_cpu = time.time() - t_start

t_start = time.time()
z2 = nm.softmax_dlayer_in_nsum_cpu(layer_out)
t_cpu2 = time.time() - t_start

print t_cpu, t_cpu2, t_cpu/t_cpu2, np.isclose(z,z2).sum()/np.single(np.prod(z.shape))