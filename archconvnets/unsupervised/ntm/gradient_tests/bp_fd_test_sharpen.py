import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

gamma = np.abs(np.random.normal(size=(8,1)))
w = np.abs(np.random.normal(size=(8, 15)))

t = np.random.normal(size=(8, 15))

def sharpen(w, gamma):
	wg = w ** gamma
	return wg / wg.sum(1)[:,np.newaxis]

def dsharpen_dgamma(w, gamma):
	n = w.shape[0]
	g = np.zeros(np.concatenate((w.shape, gamma.shape)))
	
	wg = w ** gamma
	ln_w_wg = np.log(w)*wg
	wg_sum = wg.sum(1)[:,np.newaxis]
	ln_w_wg_sum = ln_w_wg.sum(1)[:,np.newaxis]
	
	t = (ln_w_wg * wg_sum - wg * ln_w_wg_sum) / (wg_sum ** 2)
	
	g[range(n),:,range(n)] = t[:,:,np.newaxis]
	
	return g
	
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

def f(y):
	#w[i_ind,j_ind] = y
	gamma[i_ind] = y
	
	o = sharpen(w, gamma)
	
	return ((o - t)**2).sum()


def g(y):
	#w[i_ind,j_ind] = y
	gamma[i_ind] = y
	
	o = sharpen(w, gamma)
	
	##
	derr_do = sq_points_dinput(o - t)
	
	do_dg = dsharpen_dgamma(w, gamma)
	do_dw = dsharpen_dw(w, gamma)
	
	derr_dg = mult_partials_collapse(derr_do, do_dg, o)
	derr_dw = mult_partials_collapse(derr_do, do_dw, o)
	
	##
	
	return derr_dg[i_ind]
	#return derr_dw[i_ind,j_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = gamma
	#ref = w
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	#y = .2+ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	y = .2+ref[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
