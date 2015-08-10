import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

C = 4
M = 5
n_in = 3
n_shifts = 3
mem_length = 8

SCALE = .6

o_content = np.random.normal(size=(4, 5))

t = np.random.normal(size=(C, M))

def f(y):
	o_content[i_ind,j_ind] = y
	
	o = softmax(o_content)
	
	return ((o - t)**2).sum()


def g(y):
	o_content[i_ind,j_ind] = y
	
	o = softmax(o_content)
	
	derr_do = sq_points_dinput(o - t)
	do_do_content = softmax_dlayer_in_nsum(o)
	derr_do_content = mult_partials_collapse(derr_do, do_do_content, o)
	
	return derr_do_content[i_ind,j_ind]

	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = o_content
	
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
