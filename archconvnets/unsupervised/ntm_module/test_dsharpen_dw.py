from archconvnets.unsupervised.ntm.ntm_gradients import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
import numpy as np
import time

def dsharpen_dw(w, gamma):
	n = w.shape[0]
	g = np.zeros(np.concatenate((w.shape, w.shape)))

	wg = w ** gamma
	wg_sum = wg.sum(1)[:,np.newaxis]
	wg_sum2 = wg_sum ** 2
	g_wgm1 = gamma * (w ** (gamma-1))

	t = (g_wgm1 / wg_sum2) * (wg_sum - wg)

	for i in range(w.shape[0]):
		g[i,:,i,:] = t[i]
	
	for j in range(w.shape[1]):
		for b in range(w.shape[1]):
			if b != j:
				g[range(n),j,range(n),b] = -g_wgm1[:,b] * wg[:,j] / np.squeeze(wg_sum2)
	return g


w = np.asarray(np.random.random((16,6)),dtype='single')
gamma = np.asarray(np.random.random((16,1)),dtype='single')

###########
t_start = time.time()
z3 = dsharpen_dw(w, gamma)
t_cpu = time.time() - t_start

#######
t_start = time.time()
z34 = nm.dsharpen_dw_cpu(w, gamma)
t_cpu2 = time.time() - t_start

###
#nm.set_buffer(layer_out,1)

#############
#t_start = time.time()
#nm.softmax_dlayer_in_nsum(1,layer_out.shape, 2)
#z3g = nm.return_buffer(2).reshape(z3.shape)
#t_gpu = time.time() - t_start

print t_cpu, t_cpu2, t_cpu/t_cpu2, np.isclose(z3,z34).sum()/np.single(np.prod(z3.shape))
#print t_cpu2, t_gpu, t_cpu2/t_gpu, np.isclose(z3,z3g).sum()/np.single(np.prod(z3.shape))

