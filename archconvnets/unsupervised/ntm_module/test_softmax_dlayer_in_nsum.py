from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

layer_out = np.asarray(np.random.random((6,8)),dtype='single')

###########
t_start = time.time()
z3 = softmax_dlayer_in_nsum(layer_out)
t_cpu = time.time() - t_start

#######
t_start = time.time()
z34 = nm.softmax_dlayer_in_nsum_cpu(layer_out)
t_cpu2 = time.time() - t_start

###
nm.set_buffer(layer_out,1)

#############
t_start = time.time()
nm.softmax_dlayer_in_nsum(1,layer_out.shape, 2)
z3g = nm.return_buffer(2).reshape(z3.shape)
t_gpu = time.time() - t_start

print t_cpu, t_cpu2, t_cpu/t_cpu2, np.isclose(z3,z34).sum()/np.single(np.prod(z3.shape))
print t_cpu2, t_gpu, t_cpu2/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

