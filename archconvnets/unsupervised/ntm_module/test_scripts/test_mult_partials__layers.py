import archconvnets.unsupervised.ntm_module.ntm_module as nm
from archconvnets.unsupervised.ntm.ntm_gradients import *
import numpy as np
import time

OR = [None]; F = 0
DOR_DWR = [None] * 3
DOR_DWR_G = [None] * 3

derr_dor = np.asarray(np.random.random((16,6)),dtype='single')
OR[F] = np.asarray(np.random.random((16,6)),dtype='single')

DOR_DWR[0] = np.asarray(np.random.random((16,6,16,9)),dtype='single')
DOR_DWR[1] = np.asarray(np.random.random((16,6,16,3,9)),dtype='single')
DOR_DWR[2] = np.asarray(np.random.random((16,6,16,8,9)),dtype='single')

t_start = time.time()
z3 = mult_partials__layers(derr_dor, DOR_DWR, OR[F]) # 18.3%
t_cpu = time.time() - t_start

DERR_DOR = [0, derr_dor.shape]
L_OUT_BUFFER = [None] * 3

nm.set_buffer(derr_dor, DERR_DOR[0])

for i in range(3):
	DOR_DWR_G[i] = [i + 1, DOR_DWR[i].shape]
	L_OUT_BUFFER[i] = [i + 4, None]
	
	nm.set_buffer(DOR_DWR[i], DOR_DWR_G[i][0])

t_start = time.time()
nm.mult_partials__layers(DERR_DOR, DOR_DWR_G, OR[F].ndim, L_OUT_BUFFER)

for i in range(3):
	z3g = nm.return_buffer(L_OUT_BUFFER[i])
	print np.isclose(z3[i], z3g.reshape(z3[i].shape)).sum()/np.single(np.prod(z3[i].shape))

t_gpu = time.time() - t_start
print t_cpu, t_gpu, t_cpu/t_gpu
