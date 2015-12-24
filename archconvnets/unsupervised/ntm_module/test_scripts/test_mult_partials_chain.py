import archconvnets.unsupervised.ntm_module.ntm_module as nm
from archconvnets.unsupervised.ntm.ntm_gradients import *
import numpy as np
import time

#derr_dg1above = mult_partials_chain((derr_dg2above, dg2above_dg1above_relu, dg1above_relu_dg1above), (OABOVE[F_ABOVE], OABOVE[L1_ABOVE]))
	
#print derr_dg2above.shape, dg2above_dg1above_relu.shape, dg1above_relu_dg1above.shape
#print OABOVE[F_ABOVE].shape, OABOVE[L1_ABOVE].shape
##################################

OABOVE = [None]*2; F_ABOVE = 0; L1_ABOVE = 1

derr_dg2above = np.asarray(np.random.random(1),dtype='single')
dg2above_dg1above_relu = np.asarray(np.random.random((1,1,13,1)),dtype='single')
dg1above_relu_dg1above = np.asarray(np.random.random((13,1,13,1)),dtype='single')

OABOVE[F_ABOVE] = np.asarray(np.random.random((1,1)),dtype='single')
OABOVE[L1_ABOVE] = np.asarray(np.random.random((13,1)),dtype='single')

t_start = time.time()
z3 = mult_partials_chain((derr_dg2above, dg2above_dg1above_relu, dg1above_relu_dg1above), (OABOVE[F_ABOVE], OABOVE[L1_ABOVE]))
t_cpu = time.time() - t_start

########
DERR_DG2ABOVE = [0, derr_dg2above.shape]
DG2ABOVE_DG1ABOVE_RELU = [1, dg2above_dg1above_relu.shape]
DG1ABOVE_RELU_DG1ABOVE = [2, dg1above_relu_dg1above.shape]

OUT_BUFFER = [None]*3
OUT_BUFFER[0] = [3, None]
OUT_BUFFER[1] = [4, None]
OUT_BUFFER[2] = [5, None]

nm.set_buffer(derr_dg2above, DERR_DG2ABOVE[0])
nm.set_buffer(dg2above_dg1above_relu, DG2ABOVE_DG1ABOVE_RELU[0])
nm.set_buffer(dg1above_relu_dg1above, DG1ABOVE_RELU_DG1ABOVE[0])

DA_DB = [DERR_DG2ABOVE, DG2ABOVE_DG1ABOVE_RELU, DG1ABOVE_RELU_DG1ABOVE]
B_NDIM = (OABOVE[F_ABOVE].ndim, OABOVE[L1_ABOVE].ndim)

t_gpu = time.time() - t_start
nm.mult_partials_chain(DA_DB, B_NDIM, OUT_BUFFER)

z3g = nm.return_buffer(OUT_BUFFER[-1])

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
