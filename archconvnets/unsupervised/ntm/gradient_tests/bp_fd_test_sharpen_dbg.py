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
	
	g_temp = np.log(w) * w ** gamma
	
	for i in range(w.shape[0]):
		for j in range(w.shape[1]):
			g[i,j,i] = g_temp[i,j]
	
	return g

def dwg_dw(w, gamma):
	g = np.zeros(np.concatenate((w.shape, w.shape)))
	
	g_temp = gamma * w ** (gamma-1)
	
	for i in range(w.shape[0]):
		for j in range(w.shape[1]):
			g[i,j,i,j] = g_temp[i,j]
	
	return g

def f(y):
	#w[i_ind,j_ind] = y
	gamma[i_ind] = y
	
	#o = sharpen(w, gamma)
	o = w ** gamma
	
	return ((o - t)**2).sum()


def g(y):
	#w[i_ind,j_ind] = y
	gamma[i_ind] = y
	
	#o = sharpen(gamma, w)
	o = w ** gamma
	
	##
	derr_do = sq_points_dinput(o - t)
	
	do_dw = dwg_dw(w, gamma)
	do_dg = dwg_dg(w, gamma)
	
	derr_dw = mult_partials_collapse(derr_do, do_dw, o)
	derr_dg = mult_partials_collapse(derr_do, do_dg, o)
	
	##
	
	return derr_dg[i_ind]
	#return derr_dw[i_ind,j_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = gamma
	i_ind = np.random.randint(ref.shape[0])
	#j_ind = np.random.randint(ref.shape[1])
	#y = .2+ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	y = .2+ref[i_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
