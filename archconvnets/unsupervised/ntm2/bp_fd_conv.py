import numpy as np
import time
import scipy.optimize
from ntm_core import *
from model_architecture import init_model

free_all_buffers()

################ init weights and inputs
IMGS = init_buffer(np.asarray(np.random.random((1,4,32,32)),dtype='single'))
F = init_buffer(np.asarray(np.random.random((5,4,3,3)),dtype='single'))

def f(y):
	#Wy = return_buffer(IMGS)
	Wy = return_buffer(F)
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	#set_buffer(Wy.reshape(weights_shape), IMGS)
	set_buffer(Wy.reshape(weights_shape), F)
	
	O = conv((F,IMGS))
	
	z = return_buffer(O).sum()#[0,4,3,18]
	
	free_buffer(O)
	
	return z

def g(y):
	#Wy = return_buffer(IMGS)
	Wy = return_buffer(F)
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	#set_buffer(Wy.reshape(weights_shape), IMGS)
	set_buffer(Wy.reshape(weights_shape), F)
	
	O = conv((F,IMGS))
	
	DIMGS = conv_ddata((F,IMGS), O)
	DF = conv_dfilter((F,IMGS), O)
	
	#z = return_buffer(DIMGS)
	z = return_buffer(DF)
	
	free_buffer(O)
	free_buffer(DF)
	free_buffer(DIMGS)
	
	return z.sum(0).sum(0).sum(0).sum(0).ravel()[i_ind]

#ref = return_buffer(IMGS)
ref = return_buffer(F)
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e7#6

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
