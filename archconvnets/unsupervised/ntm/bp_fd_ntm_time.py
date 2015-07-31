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
#DERIV_L = SHIFT
DERIV_L = ADD
gradient_category = 'write'
#gradient_category = 'read'
####
if gradient_category == 'read':
	ref = WR[DERIV_L]
else:
	ref = WW[DERIV_L]

########
def weight_address(W, o_prev, x_cur, mem_prev):
	O = [None]*(len(W) + 5)
	
	# content
	O[KEY] = linear_2d_F(W[KEY], x_cur)
	O[CONTENT] = cosine_sim(O[KEY], mem_prev)
	
	# interpolate
	O[L1] = sq_F(W[L1], x_cur)
	O[L2] = sq_F(W[L2], O[L1])
	O[L3] = sq_F(W[L3], O[L2])
	O[IN] = interpolate(O[L3], O[CONTENT], o_prev)
	
	O[SQ] = sq_points(O[IN])
	
	# shift
	O[SHIFT] = linear_2d_F(W[SHIFT], x_cur)
	O[F] = shift_w(O[SHIFT], O[SQ])
	
	return O

def forward_pass(WR,WW, or_prev, ow_prev, mem_prev,x_cur):
	OR = weight_address(WR, or_prev, x_cur, mem_prev)
	OW = weight_address(WW, ow_prev, x_cur, mem_prev)
	
	OW[ADD] = linear_2d_F(WW[ADD], x_cur)
	
	read_mem = linear_F(OR[F], mem_prev)
	mem = mem_prev + add_mem(OW[F], OW[ADD])
	
	return OR,OW,mem,read_mem

##########
def do_dw__inputs(W, o_prev, x_cur, DO_DW, O, mem_prev, do_do_in):
	DO_DW_NEW = copy.deepcopy(DO_DW)
	
	# shift
	do_dgshift = shift_w_dshift_out_nsum(O[SQ])
	dgshift_dwshift = linear_2d_F_dF_nsum(W[SHIFT], x_cur)
	DO_DW_NEW[SHIFT] += mult_partials(do_dgshift, dgshift_dwshift, O[SHIFT])
	
	# w3
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
	
	# interp. gradients (wrt o_prev; g3)
	do_in_dg3 = interpolate_dinterp_gate_out(O[L3], O[CONTENT], o_prev)
	
	do_in_dw3 = mult_partials(do_in_dg3, dg3_dw3[:,np.newaxis], O[L3])
	do_in_dw2 = mult_partials(do_in_dg3, dg3_dw2[:,np.newaxis], O[L2])
	do_in_dw1 = mult_partials(do_in_dg3, dg3_dw1[:,np.newaxis], O[L1])

	DO_DW_NEW[L3] += mult_partials(do_do_in, do_in_dw3, O[IN])
	DO_DW_NEW[L2] += mult_partials(do_do_in, do_in_dw2, O[IN])
	DO_DW_NEW[L1] += mult_partials(do_do_in, do_in_dw1, O[IN])
	
	# interp. gradients (wrt o_content)
	do_in_do_content = interpolate_do_content(O[L3], O[CONTENT])
	do_content_dgkey = cosine_sim_expand_dkeys(O[KEY], mem_prev)
	dgkey_dwkey = linear_2d_F_dF_nsum(W[KEY], x_cur)
	
	do_content_dwkey = mult_partials(do_content_dgkey, dgkey_dwkey, O[KEY])
	do_in_dwkey = mult_partials(do_in_do_content, do_content_dwkey, O[CONTENT])
	
	DO_DW_NEW[KEY] += mult_partials(do_do_in, do_in_dwkey, O[IN])
	
	return DO_DW_NEW

def do_dw__o_prev(W, o_prev, DO_DW, O, do_do_in):
	do_in_do_prev = interpolate_do_prev(O[L3], o_prev)
	do_do_prev = mult_partials(do_do_in, do_in_do_prev, O[IN])
	
	return mult_partials__layers(do_do_prev, DO_DW, o_prev)

def do_dw__mem_prev(W, DO_DW, O, mem_prev, DMEM_PREV_DWW, do_do_in):
	do_in_do_content = interpolate_do_content(O[L3], O[CONTENT])
	do_content_dmem_prev = cosine_sim_expand_dmem(O[KEY], mem_prev)
	do_in_dmem_prev = mult_partials(do_in_do_content, do_content_dmem_prev, O[CONTENT])
	do_dmem_prev = mult_partials(do_do_in, do_in_dmem_prev, O[IN])
	
	return mult_partials__layers(do_dmem_prev, DMEM_PREV_DWW, mem_prev, DO_DW)

def mem_partials(DMEM_PREV_DWW, DOW_DWW, OW_PREV, x_prev, WW):
	DMEM_PREV_DWW_NEW = copy.deepcopy(DMEM_PREV_DWW)
	
	# write gradients
	da_dow = add_mem_dgw(OW_PREV[ADD])
	DMEM_PREV_DWW_NEW = mult_partials__layers(da_dow, DOW_DWW, OW_PREV[F], DMEM_PREV_DWW_NEW) # da_dlayer
	
	# 'add' gradients
	da_dadd_out = add_mem_dadd_out(OW_PREV[F])
	dadd_out_dwadd = linear_2d_F_dF_nsum(WW[ADD], x_prev)
	DMEM_PREV_DWW_NEW[ADD] += mult_partials(da_dadd_out, dadd_out_dwadd, OW_PREV[ADD]) # da_dwadd
	
	return DMEM_PREV_DWW_NEW

########
def f(y):
	if ref.ndim == 2 and gradient_category == 'read':
		WR[DERIV_L][i_ind,j_ind] = y
	elif gradient_category == 'read':
		WR[DERIV_L][i_ind,j_ind,k_ind] = y
	elif ref.ndim == 2:
		WW[DERIV_L][i_ind,j_ind] = y
	else:
		WW[DERIV_L][i_ind,j_ind,k_ind] = y
	##
	
	OR_PREV = copy.deepcopy(OR_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	mem_prev = copy.deepcopy(mem_previ)
	
	for frame in range(1,N_FRAMES+1):
		OR_PREV, OW_PREV, mem_prev, read_mem = forward_pass(WR, WW, OR_PREV[F], OW_PREV[F], mem_prev, x[frame])
	
	return ((read_mem - t)**2).sum()


def g(y):
	if ref.ndim == 2 and gradient_category == 'read':
		WR[DERIV_L][i_ind,j_ind] = y
	elif gradient_category == 'read':
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
	
	###
	for frame in range(1,N_FRAMES+1):
		# forward
		OR, OW, mem, read_mem = forward_pass(WR, WW, OR_PREV[F], OW_PREV[F], mem_prev, x[frame])
		
		# reverse
		dor_dor_sq = shift_w_dw_interp_nsum(OR[SHIFT])
		dow_prev_dow_prev_sq = shift_w_dw_interp_nsum(OW_PREV[SHIFT])
		
		dor_sq_dor_in = sq_points_dinput(OR[IN])
		dow_prev_sq_dow_prev_in = sq_points_dinput(OW_PREV[IN])
		
		dor_dor_in = mult_partials(dor_dor_sq, dor_sq_dor_in, OR[SQ])
		dow_prev_dow_prev_in = mult_partials(dow_prev_dow_prev_sq, dow_prev_sq_dow_prev_in, OW_PREV[SQ])
		
		# partials for weight addresses/mem
		if frame > 1:
			DOW_DWW = do_dw__o_prev(WW, OW_PREV_PREV[F], DOW_DWW, OW_PREV, dow_prev_dow_prev_in)
			DOW_DWW = do_dw__mem_prev(WW, DOW_DWW, OW_PREV, mem_prev_prev, DMEM_PREV_DWW, dow_prev_dow_prev_in)
			DOW_DWW = do_dw__inputs(WW, OW_PREV_PREV[F], x[frame-1], DOW_DWW, OW_PREV, mem_prev_prev, dow_prev_dow_prev_in)
			
			DMEM_PREV_DWW = mem_partials(DMEM_PREV_DWW, DOW_DWW, OW_PREV, x[frame-1], WW)
		
		DOR_DWR = do_dw__o_prev(WR, OR_PREV[F], DOR_DWR, OR, dor_dor_in)
		DOR_DWR = do_dw__inputs(WR, OR_PREV[F], x[frame], DOR_DWR, OR, mem_prev, dor_dor_in)
		
		DOR_DWW = do_dw__o_prev(WR, OR_PREV[F], DOR_DWW, OR, dor_dor_in)
		DOR_DWW = do_dw__mem_prev(WR, DOR_DWW, OR, mem_prev, DMEM_PREV_DWW, dor_dor_in)
	
		# update temporal state vars
		if frame != N_FRAMES:
			OW_PREV_PREV = copy.deepcopy(OW_PREV)
			OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW)
			mem_prev_prev = copy.deepcopy(mem_prev); mem_prev = copy.deepcopy(mem)
	
	########
	## full gradients:
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	# read weights
	dread_mem_dor = linear_F_dF_nsum(mem_prev)
	derr_dor = mult_partials(derr_dread_mem, dread_mem_dor, read_mem)
	
	DWR = mult_partials_collapse__layers(derr_dor, DOR_DWR, OR[F])
	DWW = mult_partials_collapse__layers(derr_dor, DOR_DWW, OR[F])
	
	# write weights
	dread_mem_dmem_prev = linear_F_dx_nsum(OR[F])
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	DWW = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DWW, mem_prev, DWW)
	
	####
	if ref.ndim == 2 and gradient_category == 'read':
		return DWR[DERIV_L][i_ind,j_ind]
	elif gradient_category == 'read':
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
