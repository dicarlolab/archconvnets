from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

#do_content_dgkey = cosine_sim_expand_dkeys(O[KEY], mem_prev) # 12.3%
mem_prev = np.asarray(np.random.random((6,8)),dtype='single')

O = [None]; KEY = 0
O_G = [None]
O[KEY] = np.asarray(np.random.random((16,8)),dtype='single')

############
t_start = time.time()
z3 = cosine_sim_expand_dkeys(O[KEY], mem_prev)
t_cpu = time.time() - t_start

####
O_G[KEY] = [1, O[KEY].shape]
MEM_PREV = [2, mem_prev.shape]
OUT_BUFFER = [3, None]

nm.set_buffer(O[KEY], O_G[KEY][0])
nm.set_buffer(mem_prev, MEM_PREV[0])


##############
t_start = time.time()
nm.cosine_sim_expand_dkeys(O_G[KEY], MEM_PREV, OUT_BUFFER)
z3g = nm.return_buffer(OUT_BUFFER)
t_gpu = time.time() - t_start

print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))