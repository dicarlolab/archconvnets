import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars2 import *

##### which gradients to test
DERIV_L = L3
read_gradients = False
#read_gradients = True
####
if read_gradients == True:
	ref = WW[DERIV_L]
else:
	ref = WR[DERIV_L]

O_KEY = np.random.normal(size=(4,5,5*8))
########
def weight_address(W, o_prev, mem_prev): # todo: shift_out, o_content computations
	O = [None]*(len(W) + 4)
	
	O[CONTENT] = linear_2d_F(O_KEY, mem_prev.ravel())
	O[F] = interpolate(W[L3], O[CONTENT], o_prev)
	
	return O

def forward_pass(WR,WW, or_prev, ow_prev, mem_prev):
	OR = weight_address(WR, or_prev, mem_prev)
	OW = weight_address(WW, ow_prev, mem_previ)
	
	read_mem = linear_F(OR[F], mem_prev)
	mem = mem_prev + add_mem(OW[F], add_out)
	
	return OR,OW,mem,read_mem

##########
def weight_address_partials(W, o_prev, DO_DW, O, mem_prev, DMEM_PREV_DWW=None):
	DO_IN_DW = [None] * len(DO_DW)
	
	# gradients of 'o' from prior time-points:
	do_in_do_prev = interpolate_do_prev(W[L3], o_prev)
	DO_IN_DW[L3] = mult_partials(do_in_do_prev, DO_DW[L3], o_prev)
	
	# w3
	DO_IN_DW[L3] += interpolate_dinterp_gate_out(W[L3], O[CONTENT], o_prev)
	
	return DO_IN_DW

def weight_address_partials_mem(W, o_prev, DO_DW, O, mem_prev, DMEM_PREV_DWW):
	DO_IN_DW = [None] * len(DO_DW)
	
	# gradients of 'o' from prior time-points:
	do_in_do_prev = interpolate_do_prev(W[L3], o_prev)
	DO_IN_DW[L3] = mult_partials(do_in_do_prev, DO_DW[L3], o_prev)
	#DO_IN_DW[L3] = np.zeros_like(mult_partials(do_in_do_prev, DO_DW[L3], o_prev))
	
	# gradients of mem from prior time-points:
	do_in_do_content = interpolate_do_content(W[L3], O[CONTENT])
	do_content_dmem_prev = linear_2d_F_dx_nsum(O_KEY).reshape((4,5,5,8))
	do_content_dlayer = mult_partials(do_content_dmem_prev, DMEM_PREV_DWW[L3], mem_prev)
	DO_IN_DW[L3] += mult_partials(do_in_do_content, do_content_dlayer, O[CONTENT])
	#DO_IN_DW[L3] += np.zeros_like(mult_partials(do_in_do_content, do_content_dlayer, O[CONTENT]))
	
	return DO_IN_DW	

def mem_partials(add_out, DMEM_PREV_DWW, DOW_DWW, OW_PREV):
	DMEM_PREV_DWW_NEW = copy.deepcopy(DMEM_PREV_DWW)
	
	da_dow = add_mem_dgw(add_out)
	
	da_dw3 = mult_partials(da_dow, DOW_DWW[L3], OW_PREV[F])
	DMEM_PREV_DWW_NEW[L3] = DMEM_PREV_DWW[L3] + da_dw3
	
	return DMEM_PREV_DWW_NEW

########
def f(y):
	if ref.ndim == 2 and read_gradients == True:
		WR[DERIV_L][i_ind,j_ind] = y
	elif read_gradients == True:
		WR[DERIV_L][i_ind,j_ind,k_ind] = y
	elif ref.ndim == 2:
		WW[DERIV_L][i_ind,j_ind] = y
	else:
		WW[DERIV_L][i_ind,j_ind,k_ind] = y
	##
	
	OR_PREV = copy.deepcopy(OR_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	mem_prev = copy.deepcopy(mem_previ)
	
	for frame in range(1,N_FRAMES+1):
		OR_PREV, OW_PREV, mem_prev, read_mem = forward_pass(WR, WW, OR_PREV[F], OW_PREV[F], mem_prev)
	
	return ((read_mem - t)**2).sum()


def g(y):
	if ref.ndim == 2 and read_gradients == True:
		WR[DERIV_L][i_ind,j_ind] = y
	elif read_gradients == True:
		WR[DERIV_L][i_ind,j_ind,k_ind] = y
	elif ref.ndim == 2:
		WW[DERIV_L][i_ind,j_ind] = y
	else:
		WW[DERIV_L][i_ind,j_ind,k_ind] = y
	##
	
	OR_PREV = copy.deepcopy(OR_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	OW_PREV_PREV = copy.deepcopy(OW_PREV_PREVi)
	DOR_DWR = copy.deepcopy(DOR_DWRi); DOW_DWW = copy.deepcopy(DOW_DWWi); DOR_DWW = copy.deepcopy(DOR_DWWi)
	mem_prev = copy.deepcopy(mem_previ); mem_prev_prev = copy.deepcopy(mem_previ)
	DMEM_PREV_DWW = copy.deepcopy(DMEM_PREV_DWWi)
	
	for frame in range(1,N_FRAMES+1):
		# forward
		OR, OW, mem, read_mem = forward_pass(WR, WW, OR_PREV[F], OW_PREV[F], mem_prev)
		
		DOW_DWW = weight_address_partials(WW, OW_PREV_PREV[F], DOW_DWW, OW_PREV, mem_prev_prev, DMEM_PREV_DWW)
		DMEM_PREV_DWW = mem_partials(add_out, DMEM_PREV_DWW, DOW_DWW, OW_PREV)
		DOR_DWR = weight_address_partials(WR, OR_PREV[F], DOR_DWR, OR, mem_prev)
		DOR_DWW = weight_address_partials_mem(WR, OR_PREV[F], DOR_DWW, OR, mem_prev, DMEM_PREV_DWW)
		
		# update temporal state vars
		if frame != N_FRAMES:
			OW_PREV_PREV = copy.deepcopy(OW_PREV)
			OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW)
			mem_prev_prev = copy.deepcopy(mem_prev)
			mem_prev = copy.deepcopy(mem)
			
	
	########
	## full gradients:
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	# read weights
	dread_mem_do = linear_F_dF_nsum(mem_prev)
	derr_do = mult_partials(derr_dread_mem, dread_mem_do, read_mem)
	
	DWR[L3] = mult_partials_sum(derr_do, DOR_DWR[L3], OR[F])
	DWW[L3] = mult_partials_sum(derr_do, DOR_DWW[L3], OR[F])
	#print DWW[L3].max(), DOR_DWW[L3].max()
	
	# write weights
	dread_mem_dmem_prev = linear_F_dx_nsum(OR[F])
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	DWW[L3] += mult_partials_sum(derr_dmem_prev, DMEM_PREV_DWW[L3], mem_prev)
	
	####
	if ref.ndim == 2 and read_gradients == True:
		return DWR[DERIV_L][i_ind,j_ind]
	elif read_gradients == True:
		return DWR[DERIV_L][i_ind,j_ind,k_ind]
	elif ref.ndim == 2:
		return DWW[DERIV_L][i_ind,j_ind]
	else:
		return DWW[DERIV_L][i_ind,j_ind,k_ind]
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e0

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):
	if ref.ndim == 2:
		i_ind = np.random.randint(ref.shape[0])
		j_ind = np.random.randint(ref.shape[1])
		y = -1e0*ref[i_ind,j_ind]
	else:
		i_ind = np.random.randint(ref.shape[0])
		j_ind = np.random.randint(ref.shape[1])
		k_ind = np.random.randint(ref.shape[2])
		y = -1e0*ref[i_ind,j_ind,k_ind]
	
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
