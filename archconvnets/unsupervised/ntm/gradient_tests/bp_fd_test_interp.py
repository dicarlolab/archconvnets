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

o_previ = np.random.normal(size=(C, M))
o_contenti = np.random.normal(size=(C, M))
o_gatei = np.random.normal(size=(C, 1))

t = np.random.normal(size=(C, M))

################# interpolate full
def interpolate(interp_gate_out, o_content, o_prev):
	return interp_gate_out * o_content + (1 - interp_gate_out) * o_prev

def interpolate_dinterp_gate_out(interp_gate_out, o_content, o_prev):
	temp = o_content - o_prev
	temp2 = np.zeros((temp.shape[0], temp.shape[1], interp_gate_out.shape[0], 1))
	
	for i in range(temp2.shape[0]):
		for j in range(temp2.shape[1]):
			temp2[i,j,i] = temp[i,j]
	return temp2

def interpolate_do_content(interp_gate_out, o_content):
	temp = interp_gate_out
	n = o_previ.shape[1]
	temp2 = np.zeros((o_previ.shape[0], n, o_previ.shape[0], n))
	
	for i in range(temp2.shape[0]):
		temp2[i,range(n),i,range(n)] = temp[i]
	return temp2

def interpolate_do_prev(o_gatei, o_previ):
	temp = 1 - o_gatei
	n = o_previ.shape[1]
	temp2 = np.zeros((o_previ.shape[0], n, o_previ.shape[0], n))
	
	for i in range(temp2.shape[0]):
		temp2[i,range(n),i,range(n)] = temp[i]
	return temp2
	
###################

def f(y):
	#o_gatei[i_ind] = y
	#o_contenti[i_ind,j_ind] = y
	o_previ[i_ind,j_ind] = y
	
	o = interpolate(o_gatei, o_contenti, o_previ)
	
	return ((o - t)**2).sum()


def g(y):
	#o_gatei[i_ind] = y
	#o_contenti[i_ind,j_ind] = y
	o_previ[i_ind,j_ind] = y
	
	o = interpolate(o_gatei, o_contenti, o_previ)
	
	derr_do = sq_points_dinput(o - t)
	
	do_do_gatei = interpolate_dinterp_gate_out(o_gatei, o_contenti, o_previ)
	do_do_contenti = interpolate_do_content(o_gatei, o_contenti)
	do_do_previ = interpolate_do_prev(o_gatei, o_previ)
	
	do_gatei = mult_partials_sum(derr_do, do_do_gatei, o)
	do_contenti = mult_partials_sum(derr_do, do_do_contenti, o)
	do_previ = mult_partials_sum(derr_do, do_do_previ, o)
	
	return do_previ[i_ind,j_ind]
	#return do_contenti[i_ind,j_ind]
	#return do_gatei[i_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e1

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	ref = o_previ#o_contenti
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind];
	
	#ref = o_gatei
	#i_ind = np.random.randint(ref.shape[0])
	#y = -1e0*ref[i_ind];
	
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
