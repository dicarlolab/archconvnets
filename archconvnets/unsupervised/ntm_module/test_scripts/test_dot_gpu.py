import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

dim1 = 5*5*2*2*60*2
dim2 = 2*2*40*2
dim3 = 5*5*2*2*80*2*2
z = np.asarray(np.random.random((dim1,dim2)),dtype='single')
z2 = np.asarray(np.random.random((dim2,dim3)),dtype='single')

t_start = time.time()
z3 = np.dot(z,z2)
t_cpu = time.time() - t_start

nm.set_buffer(z,1)
nm.set_buffer(z2,2)

t_start = time.time()
nm.dot(1,z.shape, 2, z2.shape, 3)
z3g = nm.return_buffer(3)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
