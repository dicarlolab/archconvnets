import archconvnets.unsupervised.ntm_module.ntm_module as nm
from archconvnets.unsupervised.ntm.ntm_gradients import *
import numpy as np
import time

OR = [None]; F = 0
DOR_DWR = [None] * 3

derr_dor = np.asarray(np.random.random((16,6)),dtype='single')
OR[F] = np.asarray(np.random.random((16,6)),dtype='single')

DOR_DWR[0] = np.asarray(np.random.random((16,6,16,9)),dtype='single')
DOR_DWR[1] = np.asarray(np.random.random((16,6,16,3,9)),dtype='single')
DOR_DWR[2] = np.asarray(np.random.random((16,6,16,8,9)),dtype='single')

t_start = time.time()
z3 = mult_partials__layers(derr_dor, DOR_DWR, OR[F]) # 18.3%
t_cpu = time.time() - t_start

derr_dor_ind = 0
DOR_DWR_IND = [None] * 3
DOR_DWR_SHAPE = [None] * 3
OUT_BUFFER_IND = [None] * 3

nm.set_buffer(derr_dor, derr_dor_ind)

for i in range(3):
	DOR_DWR_IND[i] = i + 1
	OUT_BUFFER_IND[i] = i + 4
	DOR_DWR_SHAPE[i] = DOR_DWR[i].shape
	nm.set_buffer(DOR_DWR[i], DOR_DWR_IND[i])

t_start = time.time()
nm.mult_partials__layers(derr_dor_ind, derr_dor.shape, DOR_DWR_IND, DOR_DWR_SHAPE, OR[F].ndim, OUT_BUFFER_IND)

for i in range(3):
	z3g = nm.return_buffer(OUT_BUFFER_IND[i])
	print np.isclose(z3[i], z3g.reshape(z3[i].shape)).sum()/np.single(np.prod(z3[i].shape))

t_gpu = time.time() - t_start
print t_cpu, t_gpu, t_cpu/t_gpu
