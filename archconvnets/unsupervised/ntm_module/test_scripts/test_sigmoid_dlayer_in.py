from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

layer_in = np.asarray(np.random.random((16,3)),dtype='single')

###########
t_start = time.time()
z3 = sigmoid_dlayer_in(layer_in)
t_cpu = time.time() - t_start

###
nm.set_buffer(layer_in,1)

#############
t_start = time.time()
nm.sigmoid_dlayer_in(1,layer_in.shape, 2)
z3g = nm.return_buffer(2).reshape(z3.shape)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

