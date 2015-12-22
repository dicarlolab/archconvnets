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

derr_dg2above_ind = 0
dg2above_dg1above_relu_ind = 1
dg1above_relu_dg1above_ind = 2
OUT_BUFFER_IND = (3,4,5)

nm.set_buffer(derr_dg2above, derr_dg2above_ind)
nm.set_buffer(dg2above_dg1above_relu, dg2above_dg1above_relu_ind)
nm.set_buffer(dg1above_relu_dg1above, dg1above_relu_dg1above_ind)

DA_DB_IND = (derr_dg2above_ind, dg2above_dg1above_relu_ind, dg1above_relu_dg1above_ind)
DA_DB_SHAPE = (derr_dg2above.shape, dg2above_dg1above_relu.shape, dg1above_relu_dg1above.shape)
B_NDIM = (OABOVE[F_ABOVE].ndim, OABOVE[L1_ABOVE].ndim)

t_gpu = time.time() - t_start
nm.mult_partials_chain(DA_DB_IND, DA_DB_SHAPE, B_NDIM, OUT_BUFFER_IND)

z3g = nm.return_buffer(OUT_BUFFER_IND[-1])

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))