import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

##### which gradients to test
DERIV_L = SHIFT
read_gradients = True
####
if read_gradients == True:
	ref = WW[DERIV_L]
else:
	ref = WR[DERIV_L]

########
def weight_address(W, o_prev, x_cur, o_content): # todo: shift_out, o_content computations
	O = [None]*(len(W) + 3)
	
	O[KEY] = linear_2d_F(W[KEY], x_cur)
	O[SHIFT] = linear_2d_F(W[SHIFT], x_cur)
	
	O[L1] = sq_F(W[L1], x_cur)
	O[L2] = sq_F(W[L2], O[L1])
	O[L3] = sq_F(W[L3], O[L2])
	O[IN] = interpolate_simp(o_prev, O[L3]) + interpolate_simp(o_content, O[L3])
	O[SQ] = sq_points(O[IN])
	
	O[F] = shift_w(O[SHIFT], O[SQ])
	
	return O

def forward_pass(WR,WW, or_prev, ow_prev, mem_prev,x_cur):
	OR = weight_address(WR, or_prev, x_cur, or_content)
	OW = weight_address(WW, ow_prev, x_cur, ow_content)
	
	read_mem = linear_F(OR[F], mem_prev)
	mem = mem_prev + add_mem(OW[F], add_out)
	
	return OR,OW,mem,read_mem

##########
def weight_address_partials(W, o_prev, o_content, x_cur, DO_DW, DO_CONTENT_DW, O):
	DO_DW_NEW = copy.deepcopy(DO_DW)
	
	# shift
	do_dgshift = shift_w_dshift_out_nsum(O[SQ])
	dgshift_dwshift = linear_2d_F_dF_nsum(W[SHIFT], x_cur)
	DO_DW_NEW[SHIFT] = mult_partials(do_dgshift, dgshift_dwshift, O[SHIFT])
	
	##
	do_do_sq = shift_w_dw_interp_nsum(O[SHIFT])
	do_sq_do_in = sq_points_dinput(O[IN])
	do_do_in = mult_partials(do_do_sq, do_sq_do_in, O[SQ])
	
	#w3
	dg3_dg2 = sq_dlayer_in_nsum(W[L3], O[L2])
	dg3_dw3 = sq_dF_nsum(W[L3], O[L2], O[L3])
	
	# w2
	dg2_dg1 = sq_dlayer_in_nsum(W[L2], O[L1])
	dg2_dw2 = sq_dF_nsum(W[L2], O[L1], O[L2])
	dg3_dw2 = mult_partials(dg3_dg2, dg2_dw2, np.squeeze(O[L2]))
	
	# w1:
	dg1_dw1 = sq_dF_nsum(W[L1], x_cur, O[L1])
	dg3_dg1 = mult_partials(dg3_dg2, dg2_dg1, np.squeeze(O[L2]))
	dg3_dw1 = mult_partials(dg3_dg1, dg1_dw1, np.squeeze(O[L1]))
	
	
	DO_DW_NEW[L3] = interpolate_simp_dx(dg3_dw3, DO_DW[L3], DO_CONTENT_DW[L3], O[L3], o_prev, o_content, do_do_in)
	DO_DW_NEW[L2] = interpolate_simp_dx(dg3_dw2, DO_DW[L2], DO_CONTENT_DW[L2], O[L3], o_prev, o_content, do_do_in)
	DO_DW_NEW[L1] = interpolate_simp_dx(dg3_dw1, DO_DW[L1], DO_CONTENT_DW[L1], O[L3], o_prev, o_content, do_do_in)
	
	return DO_DW_NEW

def mem_partials(add_out, DMEM_PREV_DWW, DOW_DWW, OW_PREV):
	DMEM_PREV_DWW_NEW = copy.deepcopy(DMEM_PREV_DWW)
	
	da_dow = add_mem_dgw(add_out)
	
	for layer in range(len(DOW_DWW)):
		da_dlayer = mult_partials(da_dow, DOW_DWW[layer], OW_PREV[F])
		DMEM_PREV_DWW_NEW[layer] = DMEM_PREV_DWW[layer] + da_dlayer
		
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
	mem_prev = copy.deepcopy(mem_previ); mem = np.zeros_like(mem_prev)
	
	for frame in range(1,N_FRAMES+1):
		OR_PREV, OW_PREV, mem_prev, read_mem = forward_pass(WR, WW, OR_PREV[F], OW_PREV[F], mem_prev, x[frame])
	
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
	DOR_DWR = copy.deepcopy(DOR_DWRi); DOW_DWW = copy.deepcopy(DOW_DWWi)
	mem_prev = copy.deepcopy(mem_previ); mem = np.zeros_like(mem_prev)
	DMEM_PREV_DWW = copy.deepcopy(DMEM_PREV_DWWi)
	
	for frame in range(1,N_FRAMES+1):
		# forward
		OR, OW, mem, read_mem = forward_pass(WR, WW, OR_PREV[F], OW_PREV[F], mem_prev, x[frame])
		
		# partials for weight addresses
		DOR_DWR = weight_address_partials(WR, OR_PREV[F], or_content, x[frame], DOR_DWR, DOR_CONTENT_DWR, OR)
		DOW_DWW = weight_address_partials(WW, OW_PREV_PREV[F], ow_content, x[frame-1], DOW_DWW, DOW_CONTENT_DWW, OW_PREV)
		
		# partials for mem
		DMEM_PREV_DWW = mem_partials(add_out, DMEM_PREV_DWW, DOW_DWW, OW_PREV)
		
		# update temporal state vars
		if frame != N_FRAMES:
			OW_PREV_PREV = copy.deepcopy(OW_PREV)
			OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW)
			mem_prev = copy.deepcopy(mem)
			
	
	########
	## full gradients:
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	# read weights
	dread_mem_do = linear_F_dF_nsum(mem_prev)
	derr_do = mult_partials(derr_dread_mem, dread_mem_do, read_mem)
	
	for layer in range(len(DWR)):
		DWR[layer] = mult_partials_sum(derr_do, DOR_DWR[layer], OR[F])
	
	# write weights
	dread_mem_dmem_prev = linear_F_dx_nsum(OR[F])
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	for layer in range(len(DWW)):
		DWW[layer] = mult_partials_sum(derr_dmem_prev, DMEM_PREV_DWW[layer], mem_prev)
	
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
eps = np.sqrt(np.finfo(np.float).eps)*1e1

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
		k_ind = np.random.randint(ref.shape[1])
		y = -1e0*ref[i_ind,j_ind,k_ind]
	
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
