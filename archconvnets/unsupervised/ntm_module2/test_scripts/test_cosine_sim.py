from archconvnets.unsupervised.ntm2.ntm_gradients import *
import archconvnets.unsupervised.ntm_module2.ntm_module2 as nm
import numpy as np
import time

keys = np.asarray(np.random.random((10*128,128*2)),dtype='single')
mem = np.asarray(np.random.random((4*256,128*2)),dtype='single')

def cosine_sim(args):
	assert len(args) == 2
	keys, mem = args
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	return numer / denom # [n_controllers, n_mem_slots]

############
t_start = time.time()
z3 = cosine_sim((keys, mem))
t_cpu = time.time() - t_start


####
KEYS = nm.init_buffer(keys)
MEM = nm.init_buffer(mem)


##############
t_start = time.time()
OUT_BUFFER = nm.cosine_sim((KEYS, MEM))
nm.sync()
t_gpu = time.time() - t_start
z3g = nm.return_buffer(OUT_BUFFER)


print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
