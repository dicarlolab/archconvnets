import archconvnets.unsupervised.ntm_module.ntm_module as nm
from archconvnets.unsupervised.ntm.ntm_gradients import *
import numpy as np
import time

#derr_dg2above_relu = sq_points_dinput(OABOVE[F_ABOVE] - t)

F_ABOVE = 0
OABOVE = [None]
OABOVE_G = [None]

OABOVE[F_ABOVE] = np.asarray(np.random.random((1,1)),dtype='single')
t = np.asarray(np.random.random((1,1)),dtype='single')

t_start = time.time()
z3 = sq_points_dinput(OABOVE[F_ABOVE] - t)
t_cpu = time.time() - t_start

####
OABOVE_G[F_ABOVE] = [0, OABOVE[F_ABOVE].shape]
T = [1, t.shape]
OUT_BUFFER = [2, None]

nm.set_buffer(OABOVE[F_ABOVE], OABOVE_G[F_ABOVE][0])
nm.set_buffer(t, T[0])

nm.point_wise_add(OABOVE_G[F_ABOVE], T, -1)

t_start = time.time()
nm.sq_points_dinput(OABOVE_G[F_ABOVE], OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
