from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

#do_content_dgkey = cosine_sim_expand_dkeys(O[KEY], mem_prev) # 12.3%
mem_prev = np.asarray(np.random.random((6,8)),dtype='single')

O = [None]; KEY = 0
O[KEY] = np.asarray(np.random.random((16,8)),dtype='single')

############
t_start = time.time()
z3 = cosine_sim_expand_dkeys(O[KEY], mem_prev)
t_cpu = time.time() - t_start


t_start = time.time()
z34 = nm.cosine_sim_expand_dkeys_cpu(O[KEY], mem_prev)
t_cpu2 = time.time() - t_start


####
nm.set_buffer(O[KEY],1)
nm.set_buffer(mem_prev,2)

##############
t_start = time.time()
nm.cosine_sim_expand_dkeys(1, O[KEY].shape, 2, mem_prev.shape, 3)
z3g = nm.return_buffer(3)
#nm.sync()
t_gpu = time.time() - t_start
#z3g = nm.return_buffer(3)

print t_cpu, t_cpu2, t_cpu/t_cpu2, np.isclose(z3, z34.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
print t_cpu2, t_gpu, t_cpu2/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))