from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

do_dgkey = np.asarray(np.random.random((16,6,16,8)),dtype='single')
dgkey_dwkey = np.asarray(np.random.random((16, 8, 16, 8, 9)),dtype='single')

O = [None]; KEY = 0

O[KEY] = np.asarray(np.random.random((16,8)),dtype='single')

t_start = time.time()
z3 = mult_partials(do_dgkey, dgkey_dwkey, O[KEY])
t_cpu = time.time() - t_start

##########
da_db = do_dgkey; db_dc = dgkey_dwkey; b = O[KEY]

nm.free_buffer(1); nm.free_buffer(2); nm.free_buffer(3)

a_ndim = da_db.ndim - b.ndim
c_ndim = db_dc.ndim - b.ndim

da_db_r = da_db.reshape((np.prod(da_db.shape[:a_ndim]), np.prod(da_db.shape[a_ndim:])))
db_dc_r = db_dc.reshape((np.prod(db_dc.shape[:b.ndim]), np.prod(db_dc.shape[b.ndim:])))

nm.set_buffer(da_db_r, 1)
nm.set_buffer(db_dc_r, 2)

t_start = time.time()
nm.dot(1,da_db_r.shape, 2, db_dc_r.shape, 3)
nm.sync()
t_gpu = time.time() - t_start

z3g = nm.return_buffer(3).reshape(np.concatenate((da_db.shape[:a_ndim], db_dc.shape[b.ndim:])))


print t_cpu, t_gpu, t_cpu/t_gpu, np.isclose(z3, z3g.reshape(z3.shape)).sum()/np.single(np.prod(z3.shape))
