import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

z = np.asarray(np.random.random((2*2*60*5,2*2*40)),dtype='single')
z2 = np.asarray(np.random.random((2*2*40,5*2*2*80)),dtype='single')

t_start = time.time()
z3 = np.dot(z,z2)
t_cpu = time.time() - t_start
print z3[0,1], z3[1,0], z3[1,1], time.time() - t_start

nm.set_buffer(z,1)
nm.set_buffer(z2,2)

#nm.dot_cpu(1,z.shape, 2, z2.shape)
nm.dot_gpu(1,z.shape, 2, z2.shape, 3)
