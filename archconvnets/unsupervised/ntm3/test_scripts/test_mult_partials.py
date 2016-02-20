import numpy as np
import time
from ntm_core import *

n_imgs = 50
dim_above = 75
N_F = 16

deriv_above_new = random_function((n_imgs, dim_above, N_F, 8))
p_partial = random_function((n_imgs, N_F, 8, N_F, 3))
out_shape = (n_imgs, N_F, 8)


DERIV_ABOVE_NEW = init_buffer(deriv_above_new)
P_PARTIAL = init_buffer(p_partial)

t_start = time.time()
DERIV_TEMP = mult_partials(DERIV_ABOVE_NEW, P_PARTIAL, out_shape, True)
t_gpu = time.time() - t_start

deriv_temp = return_buffer(DERIV_TEMP)

deriv_temp_target = np.zeros(deriv_temp.shape, dtype='single')

t_start = time.time()
for img in range(n_imgs):
	for batch in range(dim_above):
		deriv_temp_target[img,batch] = np.dot(deriv_above_new[img,batch].reshape((1,N_F*8)), p_partial[img].reshape((N_F*8,N_F*3))).reshape((N_F,3))
t_cpu = time.time() - t_start

print t_gpu, t_cpu, t_cpu/t_gpu
		
print np.isclose(deriv_temp, deriv_temp_target).sum()/np.single(np.prod(deriv_temp.shape))
