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

##############
#t_start = time.time()
#z3g = mult_partials_gpu(do_dgkey, dgkey_dwkey, O[KEY])
#t_gpu = time.time() - t_start

#print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
