from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

a = np.asarray(np.random.random((3,16,3,5,6)),dtype='single')
b = np.asarray(np.random.random((16,3)),dtype='single')
s = 2.3

A = [None]*3
for i in range(3):
	A[i] = nm.init_buffer(a[i])

B = nm.init_buffer(b)

###########
t_start = time.time()
for i in range(3):
	a[i] *= b[:,:,np.newaxis,np.newaxis]*s
t_cpu = time.time() - t_start

z3g = [None]*3
#############
t_start = time.time()
nm.pointwise_mult_partials__layers(A, B, scalar=s)
for i in range(3):
	z3g[i] = nm.return_buffer(A[i])
t_gpu = time.time() - t_start

for i in range(3):
	print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(a[i],z3g[i]).sum()/np.single(np.prod(a[i].shape))

