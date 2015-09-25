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

def dwg_dg(w, gamma):
	g = np.zeros(np.concatenate((w.shape, gamma.shape)))
	
	wg = w ** gamma
	ln_w_wg = np.log(w)*wg
	wg_sum = wg.sum(1)[:,np.newaxis]
	ln_w_wg_sum = ln_w_wg.sum(1)[:,np.newaxis]
	
	for i in range(w.shape[0]):
		for j in range(w.shape[1]):
			g[i,j,i] = (ln_w_wg[i,j] * wg_sum[i] - wg[i,j] * ln_w_wg_sum[i]) / (wg_sum[i]**2)
	
	return g
	
def dwg_dw(w, gamma):
	g = np.zeros(np.concatenate((w.shape, w.shape)))
	
	wg = w ** gamma
	wg_sum = wg.sum(1)[:,np.newaxis]
	wg_sum2 = wg_sum ** 2
	g_wgm1 = gamma * (w ** (gamma-1))
	
	for i in range(w.shape[0]):
		for j in range(w.shape[1]):
			g[i,j,i,j] = (g_wgm1[i,j] / wg_sum2[i]) * (wg_sum[i] - wg[i,j])
			
			for b in range(w.shape[1]):
				if b != j:
					g[i,j,i,b] = -g_wgm1[i,b] * wg[i,j] / wg_sum2[i]
	
	return g

def f(y):
	w[i_ind,j_ind] = y
	#gamma[i_ind] = y
	
	numer = (w ** gamma)
	denom = (w ** gamma).sum(1)[:,np.newaxis]
	o = numer / denom
	
	return ((o - t)**2).sum()


def g(y):
	w[i_ind,j_ind] = y
	#gamma[i_ind] = y
	
	numer = (w ** gamma)
	denom = (w ** gamma).sum(1)[:,np.newaxis]
	o = numer / denom
	
	##
	derr_do = sq_points_dinput(o - t)
	
	do_dg = dwg_dg(w, gamma)
	do_dw = dwg_dw(w, gamma)
	
	derr_dg = mult_partials_collapse(derr_do, do_dg, o)
	derr_dw = mult_partials_collapse(derr_do, do_dw, o)
	
	##
	
	#return derr_dg[i_ind]
	return derr_dw[i_ind,j_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = gamma
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = .2+ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	#y = .2+ref[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
