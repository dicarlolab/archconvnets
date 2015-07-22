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
#gradient_category = 'write'
gradient_category = 'read'
####
if gradient_category == 'add':
	ref = wadd
elif gradient_category == 'read':
	ref = WW[DERIV_L]
else:
	ref = WR[DERIV_L]

########
def weight_address(W, o_prev, x_cur, mem_prev): # todo: shift_out, o_content computations
	O = [None]*(len(W) + 4)
	
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
	
	add_out = linear_2d_F(wadd, x_cur)
	
	read_mem = linear_F(OR[F], mem_prev)
	mem = mem_prev + add_mem(OW[F], add_out)
	
	return OR,OW,mem,read_mem,add_out

##########
def weight_address_partials(W, o_prev, x_cur, DO_DW, O, mem_prev, DMEM_PREV_DWW=None):
	DO_DW_NEW = copy.deepcopy(DO_DW)
	DO_IN_DW = [None] * len(DO_DW)
	
	##
	do_do_sq = shift_w_dw_interp_nsum(O[SHIFT])
	do_sq_do_in = sq_points_dinput(O[IN])
	do_do_in = mult_partials(do_do_sq, do_sq_do_in, O[SQ])
	do_in_do_content = interpolate_do_content(O[L3], O[CONTENT])
	
	# gradients of 'o' from prior time-points:
	do_in_do_prev = interpolate_do_prev(O[L3], o_prev)
	for layer in range(len(DO_DW)):
		DO_IN_DW[layer] = mult_partials(do_in_do_prev, DO_DW[layer], o_prev)
	
	# gradients of mem from prior time-points:
	if DMEM_PREV_DWW != None:
		do_content_dmem_prev = cosine_sim_expand_dmem(O[KEY], mem_prev)
		for layer in range(len(DO_DW)):
			do_content_dlayer = mult_partials(do_content_dmem_prev, DMEM_PREV_DWW[layer], mem_prev)
			DO_IN_DW[layer] += mult_partials(do_in_do_content, do_content_dlayer, O[CONTENT])
		
	# shift
	do_dgshift = shift_w_dshift_out_nsum(O[SQ])
	dgshift_dwshift = linear_2d_F_dF_nsum(W[SHIFT], x_cur)
	DO_DW_NEW[SHIFT] = mult_partials(do_dgshift, dgshift_dwshift, O[SHIFT])
	
	#do_sq_dwshift = mult_partials(do_sq_do_in, DO_IN_DW[SHIFT], O[IN])
	#DO_DW_NEW[SHIFT] += mult_partials(do_do_sq, do_sq_dwshift, O[SQ])
	DO_DW_NEW[SHIFT] += mult_partials(do_do_in, DO_IN_DW[SHIFT], O[SQ])
	
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

	DO_DW_NEW[L3] = mult_partials(do_do_in, do_in_dw3 + DO_IN_DW[L3], O[IN])
	DO_DW_NEW[L2] = mult_partials(do_do_in, do_in_dw2 + DO_IN_DW[L2], O[IN])
	DO_DW_NEW[L1] = mult_partials(do_do_in, do_in_dw1 + DO_IN_DW[L1], O[IN])
	
	# interp. gradients (wrt o_content)
	do_content_dgkey = cosine_sim_expand_dkeys(O[KEY], mem_prev)
	dgkey_dwkey = linear_2d_F_dF_nsum(W[KEY], x_cur)
	
	do_content_dwkey = mult_partials(do_content_dgkey, dgkey_dwkey, O[KEY])
	do_in_dwkey = mult_partials(do_in_do_content, do_content_dwkey, O[CONTENT])
	
	DO_DW_NEW[KEY] = mult_partials(do_do_in, do_in_dwkey + DO_IN_DW[KEY], O[IN])
	
	return DO_DW_NEW

def read_dwrite_weight_address_partials(W, o_prev, x_cur, DO_DW, O, mem_prev, DMEM_PREV_DWW, dor_dwadd, dmem_prev_dwadd):
	DO_DW_NEW = copy.deepcopy(DO_DW)
	DO_IN_DW = [None] * len(DO_DW)
	
	##
	do_do_sq = shift_w_dw_interp_nsum(O[SHIFT])
	do_sq_do_in = sq_points_dinput(O[IN])
	do_do_in = mult_partials(do_do_sq, do_sq_do_in, O[SQ])
	
	# gradients of 'o' from prior time-points:
	do_in_do_prev = interpolate_do_prev(O[L3], o_prev)
	
	do_in_dwadd = mult_partials(do_in_do_prev, dor_dwadd, o_prev) # 'add' weight
	
	for layer in range(len(DO_DW)):
		DO_IN_DW[layer] = mult_partials(do_in_do_prev, DO_DW[layer], o_prev)
	
	
	# gradients of mem from prior time-points:
	do_in_do_content = interpolate_do_content(O[L3], O[CONTENT])
	do_content_dmem_prev = cosine_sim_expand_dmem(O[KEY], mem_prev)
	
	do_content_dwadd = mult_partials(do_content_dmem_prev, dmem_prev_dwadd, mem_prev)
	do_in_dwadd += mult_partials(do_in_do_content, do_content_dwadd, O[CONTENT])
	dor_dwadd_new = mult_partials(do_do_in, do_in_dwadd, O[IN])
	
	for layer in range(len(DO_DW)):
		do_content_dlayer = mult_partials(do_content_dmem_prev, DMEM_PREV_DWW[layer], mem_prev)
		DO_IN_DW[layer] += mult_partials(do_in_do_content, do_content_dlayer, O[CONTENT])
		DO_DW_NEW[layer] = mult_partials(do_do_in, DO_IN_DW[layer], O[IN])
	
	return DO_DW_NEW,dor_dwadd_new

def mem_partials(add_out, DMEM_PREV_DWW, DOW_DWW, OW_PREV, x_prev, dmem_prev_dwadd, dow_dwadd):
	DMEM_PREV_DWW_NEW = copy.deepcopy(DMEM_PREV_DWW)
	dmem_prev_dwadd_new = copy.deepcopy(dmem_prev_dwadd)
	
	# write gradients
	da_dow = add_mem_dgw(add_out)
	
	dmem_prev_dwadd_new += mult_partials(da_dow, dow_dwadd, OW_PREV[F]) # da_dwadd
	
	for layer in range(len(DOW_DWW)):
		DMEM_PREV_DWW_NEW[layer] += mult_partials(da_dow, DOW_DWW[layer], OW_PREV[F]) # da_dlayer
	
	# 'add' gradients
	da_dadd_out = add_mem_dadd_out(OW_PREV[F])
	dadd_out_dwadd = linear_2d_F_dF_nsum(wadd, x_prev)
	dmem_prev_dwadd_new += mult_partials(da_dadd_out, dadd_out_dwadd, add_out) # da_dwadd
	
	return DMEM_PREV_DWW_NEW, dmem_prev_dwadd_new

########
def f(y):
	if gradient_category == 'add':
		wadd[i_ind,j_ind,k_ind] = y
	elif ref.ndim == 2 and gradient_category == 'read':
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
		OR_PREV, OW_PREV, mem_prev, read_mem = forward_pass(WR, WW, OR_PREV[F], OW_PREV[F], mem_prev, x[frame])[:4]
	
	return ((read_mem - t)**2).sum()


def g(y):
	if gradient_category == 'add':
		wadd[i_ind,j_ind,k_ind] = y
	elif ref.ndim == 2 and gradient_category == 'read':
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
	dmem_prev_dwadd = copy.deepcopy(dmem_prev_dwaddi);
	dor_dwadd = copy.deepcopy(dor_dwaddi); dow_dwadd = copy.deepcopy(dow_dwaddi)
	
	###
	for frame in range(1,N_FRAMES+1):
		# forward
		OR, OW, mem, read_mem, add_out = forward_pass(WR, WW, OR_PREV[F], OW_PREV[F], mem_prev, x[frame])
		
		# partials for weight addresses/mem
		if frame > 1:
			dow_dwadd = read_dwrite_weight_address_partials(WW, OW_PREV_PREV[F], x[frame-1], DOW_DWW, OW_PREV, mem_prev_prev, DMEM_PREV_DWW, dow_dwadd, dmem_prev_dwadd)[1]
			DOW_DWW = weight_address_partials(WW, OW_PREV_PREV[F], x[frame-1], DOW_DWW, OW_PREV, mem_prev_prev, DMEM_PREV_DWW)
			
			DMEM_PREV_DWW, dmem_prev_dwadd = mem_partials(add_out_prev, DMEM_PREV_DWW, DOW_DWW, OW_PREV, x[frame-1], dmem_prev_dwadd, dow_dwadd)
		DOR_DWR = weight_address_partials(WR, OR_PREV[F], x[frame], DOR_DWR, OR, mem_prev)
		DOR_DWW, dor_dwadd = read_dwrite_weight_address_partials(WR, OR_PREV[F], x[frame], DOR_DWW, OR, mem_prev, DMEM_PREV_DWW, dor_dwadd, dmem_prev_dwadd)
		
	
		# update temporal state vars
		if frame != N_FRAMES:
			OW_PREV_PREV = copy.deepcopy(OW_PREV)
			OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW)
			mem_prev_prev = copy.deepcopy(mem_prev)
			mem_prev = copy.deepcopy(mem)
			add_out_prev = copy.deepcopy(add_out)
			
	
	########
	## full gradients:
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	# read weights
	dread_mem_dor = linear_F_dF_nsum(mem_prev)
	derr_dor = mult_partials(derr_dread_mem, dread_mem_dor, read_mem)
	
	for layer in range(len(DWR)):
		DWR[layer] = mult_partials_sum(derr_dor, DOR_DWR[layer], OR[F])
		DWW[layer] = mult_partials_sum(derr_dor, DOR_DWW[layer], OR[F])
		dwadd = mult_partials_sum(derr_dor, dor_dwadd, OR[F])
	
	# write weights
	dread_mem_dmem_prev = linear_F_dx_nsum(OR[F])
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	for layer in range(len(DWW)):
		DWW[layer] += mult_partials_sum(derr_dmem_prev, DMEM_PREV_DWW[layer], mem_prev)
		dwadd += mult_partials_sum(derr_dmem_prev, dmem_prev_dwadd, mem_prev)
	
	####
	if gradient_category == 'add':
		return dwadd[i_ind,j_ind,k_ind]
	elif ref.ndim == 2 and gradient_category == 'read':
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
