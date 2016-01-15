import numpy as np
import time
import scipy.optimize
from ntm_core import *
from model_architecture import init_model

free_all_buffers()

################ init weights and inputs
O = init_buffer(np.asarray(np.random.random((1, 5, 30, 30)),dtype='single'))

def f(y):
	Wy = return_buffer(O)
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), O)
	
	MO = max_pool([O])
	
	z = return_buffer(MO).sum()#[0,4,3,18]
	
	free_buffer(MO)
	
	return z

def g(y):
	Wy = return_buffer(O)
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), O)
	
	MO = max_pool([O])
	
	DO = max_pool_dinput([O], MO)
	
	z = return_buffer(DO)
	
	free_buffer(DO)
	free_buffer(MO)
	
	return z.sum(0).sum(0).sum(0).sum(0).ravel()[i_ind]

ref = return_buffer(O)
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e5#6

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
t_start = time.time()
for sample in range(N_SAMPLES):
	i_ind = np.random.randint(np.prod(ref.shape))
	y = ref.ravel()[i_ind]
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std(), time.time() - t_start, GPU
