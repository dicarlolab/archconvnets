from archconvnets.unsupervised.ntm2.ntm_gradients import *
import archconvnets.unsupervised.ntm_module2.ntm_module2 as nm
import numpy as np
import time

keys = np.asarray(np.random.random((12,8)),dtype='single')
mem_prev = np.asarray(np.random.random((6,8)),dtype='single')

def cosine_sim(args):
	assert len(args) == 2
	keys, mem = args
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	return numer / denom # [n_controllers, n_mem_slots]

############
t_start = time.time()
z3 = cosine_sim((keys, mem_prev))
t_cpu = time.time() - t_start

####
'''O_G[KEY] = [1, O[KEY].shape]
MEM_PREV = [2, mem_prev.shape]
OUT_BUFFER = [3, None]

nm.set_buffer(O[KEY], O_G[KEY][0])
nm.set_buffer(mem_prev, MEM_PREV[0])


##############
t_start = time.time()
nm.cosine_sim_expand_dkeys(O_G[KEY], MEM_PREV, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))'''