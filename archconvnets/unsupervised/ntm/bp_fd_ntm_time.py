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
#DERIV_L = L2_UNDER
#DERIV_L = F_UNDER
#DERIV_L = SHIFT
DERIV_L = IN_GATE
#DERIV_L = KEY
#DERIV_L = BETA ## ??
gradient_category = 'write'
#gradient_category = 'read'
#gradient_category = 'under'
####
if gradient_category == 'under':
	ref = WUNDER[DERIV_L]
elif gradient_category == 'read':
	ref = WR[DERIV_L]
else:
	ref = WW[DERIV_L]

########
def weight_address(W, O_PREV, inputs, mem_prev):
	O = [None]*len(O_PREV)
	
	# content
	O[KEY] = linear_2d_F(W[KEY], inputs)
	O[BETA] = linear_F(W[BETA], inputs)
	O[KEY_FOCUSED] = focus_keys(O[KEY], O[BETA])
	O[CONTENT] = cosine_sim(O[KEY_FOCUSED], mem_prev)
	O[CONTENT_SM] = sq_points(O[CONTENT])
	
	# interpolate
	O[IN_GATE] = linear_F(W[IN_GATE], inputs)
	O[IN] = interpolate(O[IN_GATE], O[CONTENT], O_PREV[F])
	
	O[SQ] = sq_points(O[IN])
	
	# shift
	O[SHIFT] = linear_2d_F(W[SHIFT], inputs)
	O[F] = shift_w(O[SHIFT], O[SQ])
	
	return O

def forward_pass(WUNDER, WR,WW, OR_PREV, OW_PREV, mem_prev, x_cur):
	OUNDER = [None]*len(WUNDER)
	
	# processing underneath read/write heads
	OUNDER[L1_UNDER] = sq_F(WUNDER[L1_UNDER], x_cur)
	OUNDER[L2_UNDER] = sq_F(WUNDER[L2_UNDER], OUNDER[L1_UNDER])
	OUNDER[F_UNDER] = sq_F(WUNDER[F_UNDER], OUNDER[L2_UNDER])
	
	# read/write heads
	OR = weight_address(WR, OR_PREV, OUNDER[F_UNDER], mem_prev)
	OW = weight_address(WW, OW_PREV, OUNDER[F_UNDER], mem_prev)
	
	# add output
	OW[ADD] = linear_2d_F(WW[ADD], OUNDER[F_UNDER])
	
	# read then write to mem
	read_mem = linear_F(OR[F], mem_prev)
	mem = mem_prev + add_mem(OW[F], OW[ADD])
	
	return OR,OW,mem,read_mem,OUNDER

def dunder_dw(WUNDER, OUNDER, x):
	DG3UNDER_DW = [None] * len(OUNDER)
	
	DG3UNDER_DW[F_UNDER] = sq_dF_nsum(WUNDER[F_UNDER], OUNDER[L2_UNDER], OUNDER[F_UNDER])
	
	dg3under_dg2under = sq_dlayer_in_nsum(WUNDER[F_UNDER], OUNDER[L2_UNDER])
	dg2under_dw2under = sq_dF_nsum(WUNDER[L2_UNDER], OUNDER[L1_UNDER], OUNDER[L2_UNDER])
	DG3UNDER_DW[L2_UNDER] = mult_partials(dg3under_dg2under, dg2under_dw2under,  np.squeeze(OUNDER[L2_UNDER]))
	
	dg2under_dg1under = sq_dlayer_in_nsum(WUNDER[L2_UNDER], OUNDER[L1_UNDER])
	dg1under_dw1under = sq_dF_nsum(WUNDER[L1_UNDER], x, OUNDER[L1_UNDER])
	dg2under_dw1under = mult_partials(dg2under_dg1under, dg1under_dw1under,  np.squeeze(OUNDER[L1_UNDER]))
	DG3UNDER_DW[L1_UNDER] = mult_partials(dg3under_dg2under, dg2under_dw1under, np.squeeze(OUNDER[L2_UNDER]))
	
	return DG3UNDER_DW

########## ...
def do_dw__inputs(W, WUNDER, o_prev, OUNDER, DO_DWUNDER, O, DO_DW, mem_prev, x, do_do_in):
	DO_DW_NEW = copy.deepcopy(DO_DW)
	DO_DWUNDER_NEW = copy.deepcopy(DO_DWUNDER)
	
	## shift weights
	do_dgshift = shift_w_dshift_out_nsum(O[SQ])
	dgshift_dwshift = linear_2d_F_dF_nsum(W[SHIFT], OUNDER[F_UNDER])
	dgshift_dg3under = linear_2d_F_dx_nsum(W[SHIFT])
	DO_DW_NEW[SHIFT] += mult_partials(do_dgshift, dgshift_dwshift, O[SHIFT])
	do_dg3under = mult_partials(do_dgshift, dgshift_dg3under, O[SHIFT])
	
	## interp. gradients (wrt o_prev; gin_gate)
	do_in_dgin_gate = interpolate_dinterp_gate_out(O[IN_GATE], O[CONTENT], o_prev)
	#do_in_dgin_gate = interpolate_dinterp_gate_out(O[IN_GATE], O[CONTENT_SM], o_prev)
	do_dgin_gate = mult_partials(do_do_in, do_in_dgin_gate, O[IN])
	dgin_gate_dg3under = linear_F_dx_nsum_g(W[IN_GATE], OUNDER[F_UNDER])
	dgin_gate_dwin = linear_F_dF_nsum_g(W[IN_GATE], OUNDER[F_UNDER])
	DO_DW_NEW[IN_GATE] += mult_partials(do_dgin_gate, dgin_gate_dwin, O[IN_GATE])
	do_dg3under += np.squeeze(mult_partials(do_dgin_gate, dgin_gate_dg3under, O[IN_GATE]))
	
	## interp. gradients (wrt o_content)
	do_in_do_content = interpolate_do_content(O[IN_GATE], O[CONTENT])
	#do_in_do_content_sm = interpolate_do_content(O[IN_GATE], O[CONTENT_SM])
	#do_content_sm_do_content = sq_points_dinput(O[CONTENT_SM])#softmax_dlayer_in_nsum(O[CONTENT_SM])
	#do_in_do_content = mult_partials(do_in_do_content_sm, do_content_sm_do_content, O[CONTENT_SM])
	do_content_dgkey_focused = cosine_sim_expand_dkeys(O[KEY_FOCUSED], mem_prev)
	do_in_dgkey_focused = mult_partials(do_in_do_content, do_content_dgkey_focused, O[CONTENT])
	do_dgkey_focused = mult_partials(do_do_in, do_in_dgkey_focused, O[IN])
	
	# key
	dgkey_focused_dgkey = focus_key_dkeys_nsum(O[KEY_FOCUSED], O[BETA])
	do_dgkey = mult_partials(do_dgkey_focused, dgkey_focused_dgkey, O[KEY_FOCUSED])
	dgkey_dwkey = linear_2d_F_dF_nsum(W[KEY], OUNDER[F_UNDER])
	dgkey_dg3under = linear_2d_F_dx_nsum(W[KEY])
	DO_DW_NEW[KEY] += mult_partials(do_dgkey, dgkey_dwkey, O[KEY])
	do_dg3under += mult_partials(do_dgkey, dgkey_dg3under, O[KEY])
	
	# beta
	dgkey_focused_dgbeta = focus_key_dbeta_out_nsum(O[KEY_FOCUSED], O[BETA])
	do_dgbeta = mult_partials(do_dgkey_focused, dgkey_focused_dgbeta, O[KEY_FOCUSED])
	dgbeta_dwbeta = linear_F_dF_nsum_g(W[BETA], OUNDER[F_UNDER])
	dgbeta_dg3under = linear_F_dx_nsum_g(W[BETA], OUNDER[F_UNDER])
	DO_DW_NEW[BETA] += mult_partials(do_dgbeta, dgbeta_dwbeta, O[BETA])
	do_dg3under += np.squeeze(mult_partials(do_dgbeta, dgbeta_dg3under, O[BETA]))
	
	## combine weights under gradients
	DG3UNDER_DW = dunder_dw(WUNDER, OUNDER, x)
	DO_DWUNDER_NEW = mult_partials__layers(do_dg3under, DG3UNDER_DW, np.squeeze(OUNDER[F_UNDER]), DO_DWUNDER_NEW)
	
	return DO_DW_NEW, DO_DWUNDER_NEW

########## ...
def do_dw__o_prev(W, o_prev, DO_DW, DO_DWUNDER, O, do_do_in):
	do_in_do_prev = interpolate_do_prev(O[IN_GATE], o_prev)
	do_do_prev = mult_partials(do_do_in, do_in_do_prev, O[IN])
	
	DO_DW_NEW = mult_partials__layers(do_do_prev, DO_DW, o_prev)
	DO_DWUNDER_NEW = mult_partials__layers(do_do_prev, DO_DWUNDER, o_prev)
	
	return DO_DW_NEW, DO_DWUNDER_NEW

def do_dw__mem_prev(W, DO_DW, DO_DWUNDER, O, mem_prev, DMEM_PREV_DWW, DMEM_PREV_DWUNDER, do_do_in):
	do_in_do_content = interpolate_do_content(O[IN_GATE], O[CONTENT])
	do_content_dmem_prev = cosine_sim_expand_dmem(O[KEY_FOCUSED], mem_prev)
	do_in_dmem_prev = mult_partials(do_in_do_content, do_content_dmem_prev, O[CONTENT])
	#do_in_do_content_sm = interpolate_do_content(O[IN_GATE], O[CONTENT_SM])
	#do_content_sm_dmem_prev = cosine_sim_expand_dmem(O[KEY_FOCUSED], mem_prev)
	#do_in_dmem_prev = mult_partials(do_in_do_content_sm, do_content_sm_dmem_prev, O[CONTENT_SM])
	do_dmem_prev = mult_partials(do_do_in, do_in_dmem_prev, O[IN])
	
	DO_DW_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWW, mem_prev, DO_DW)
	DO_DWUNDER_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWUNDER, mem_prev, DO_DWUNDER)
	
	return DO_DW_NEW, DO_DWUNDER_NEW

def mem_partials(DMEM_PREV_DWW, DMEM_PREV_DWUNDER, DOW_DWW, DOW_DWUNDER, OW_PREV, OUNDER_PREV, WW, WUNDER, x_prev):
	# write gradients
	da_dow = add_mem_dgw(OW_PREV[ADD])
	DMEM_PREV_DWW_NEW = mult_partials__layers(da_dow, DOW_DWW, OW_PREV[F], DMEM_PREV_DWW) # da_dlayer
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(da_dow, DOW_DWUNDER, OW_PREV[F], DMEM_PREV_DWUNDER)
	
	###
	# 'add' gradients
	da_dadd_out = add_mem_dadd_out(OW_PREV[F])
	
	dadd_out_dwadd = linear_2d_F_dF_nsum(WW[ADD], OUNDER_PREV[F_UNDER])
	DMEM_PREV_DWW_NEW[ADD] += mult_partials(da_dadd_out, dadd_out_dwadd, OW_PREV[ADD]) # da_dwadd
	
	# under:
	dadd_out_dg3under = linear_2d_F_dx_nsum(WW[ADD])
	da_dg3under = mult_partials(da_dadd_out, dadd_out_dg3under, OW_PREV[ADD])
	
	DG3UNDER_DW = dunder_dw(WUNDER, OUNDER_PREV, x_prev)
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(da_dg3under, DG3UNDER_DW, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DWUNDER_NEW)
	
	return DMEM_PREV_DWW_NEW, DMEM_PREV_DWUNDER_NEW

########
def f(y):
	if ref.ndim == 2 and gradient_category == 'under':
		WUNDER[DERIV_L][i_ind,j_ind] = y
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
		OR_PREV, OW_PREV, mem_prev, read_mem = forward_pass(WUNDER, WR, WW, OR_PREV, OW_PREV, mem_prev, x[frame])[:4]
	
	return ((read_mem - t)**2).sum()


def g(y):
	if ref.ndim == 2 and gradient_category == 'under':
		WUNDER[DERIV_L][i_ind,j_ind] = y
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
	OW_PREV_PREV = copy.deepcopy(OW_PREV_PREVi); OUNDER_PREV = copy.deepcopy(OUNDER_PREVi)
	DOR_DWR = copy.deepcopy(DOR_DWRi); DOW_DWW = copy.deepcopy(DOW_DWWi); DOR_DWW = copy.deepcopy(DOR_DWWi)
	DOW_DWUNDER = copy.deepcopy(DOW_DWUNDERi); DOR_DWUNDER = copy.deepcopy(DOR_DWUNDERi)
	mem_prev = copy.deepcopy(mem_previ); mem_prev_prev = copy.deepcopy(mem_previ)
	DMEM_PREV_DWW = copy.deepcopy(DMEM_PREV_DWWi); DMEM_PREV_DWUNDER = copy.deepcopy(DMEM_PREV_DWUNDERi)
	
	###
	for frame in range(1,N_FRAMES+1):
		# forward
		OR, OW, mem, read_mem, OUNDER = forward_pass(WUNDER, WR, WW, OR_PREV, OW_PREV, mem_prev, x[frame])
		
		# reverse
		dor_dor_sq = shift_w_dw_interp_nsum(OR[SHIFT])
		dow_prev_dow_prev_sq = shift_w_dw_interp_nsum(OW_PREV[SHIFT])
		
		dor_sq_dor_in = sq_points_dinput(OR[IN])
		dow_prev_sq_dow_prev_in = sq_points_dinput(OW_PREV[IN])
		
		dor_dor_in = mult_partials(dor_dor_sq, dor_sq_dor_in, OR[SQ])
		dow_prev_dow_prev_in = mult_partials(dow_prev_dow_prev_sq, dow_prev_sq_dow_prev_in, OW_PREV[SQ])
		
		# partials for write head output (OW)
		if frame > 1:
			DOW_DWW, DOW_DWUNDER = do_dw__o_prev(WW, OW_PREV_PREV[F], DOW_DWW, DOW_DWUNDER, OW_PREV, dow_prev_dow_prev_in)
			DOW_DWW, DOW_DWUNDER = do_dw__mem_prev(WW, DOW_DWW, DOW_DWUNDER, OW_PREV, mem_prev_prev, DMEM_PREV_DWW, DMEM_PREV_DWUNDER, dow_prev_dow_prev_in)
			DOW_DWW, DOW_DWUNDER = do_dw__inputs(WW, WUNDER, OW_PREV_PREV[F], OUNDER_PREV, DOW_DWUNDER, OW_PREV, DOW_DWW, mem_prev_prev, x[frame-1], dow_prev_dow_prev_in)
			
			DMEM_PREV_DWW, DMEM_PREV_DWUNDER = mem_partials(DMEM_PREV_DWW, DMEM_PREV_DWUNDER, DOW_DWW, DOW_DWUNDER, OW_PREV, OUNDER_PREV, WW, WUNDER, x[frame-1])
		
		# partials from read head output (OR)
		DOR_DWR, DOR_DWUNDER = do_dw__o_prev(WR, OR_PREV[F], DOR_DWR, DOR_DWUNDER, OR, dor_dor_in)
		DOR_DWR, DOR_DWUNDER = do_dw__inputs(WR, WUNDER, OR_PREV[F], OUNDER, DOR_DWUNDER, OR, DOR_DWR, mem_prev, x[frame], dor_dor_in)
		
		DOR_DWW = do_dw__o_prev(WR, OR_PREV[F], DOR_DWW, DOR_DWUNDER, OR, dor_dor_in)[0] #?
		DOR_DWW, DOR_DWUNDER = do_dw__mem_prev(WR, DOR_DWW, DOR_DWUNDER, OR, mem_prev, DMEM_PREV_DWW, DMEM_PREV_DWUNDER, dor_dor_in)
	
		# update temporal state vars
		if frame != N_FRAMES:
			OW_PREV_PREV = copy.deepcopy(OW_PREV)
			OR_PREV = copy.deepcopy(OR); OW_PREV = copy.deepcopy(OW); OUNDER_PREV = copy.deepcopy(OUNDER)
			mem_prev_prev = copy.deepcopy(mem_prev); mem_prev = copy.deepcopy(mem)
	
	########
	## full gradients:
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	# read weights
	dread_mem_dor = linear_F_dF_nsum(mem_prev)
	derr_dor = mult_partials(derr_dread_mem, dread_mem_dor, read_mem)
	
	DWR = mult_partials_collapse__layers(derr_dor, DOR_DWR, OR[F])
	DWW = mult_partials_collapse__layers(derr_dor, DOR_DWW, OR[F])
	DWUNDER = mult_partials_collapse__layers(derr_dor, DOR_DWUNDER, OR[F])
	
	# write weights
	dread_mem_dmem_prev = linear_F_dx_nsum(OR[F])
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	DWW = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DWW, mem_prev, DWW)
	DWUNDER = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DWUNDER, mem_prev, DWUNDER)
	
	
	####
	if ref.ndim == 2 and gradient_category == 'under':
		return DWUNDER[DERIV_L][i_ind,j_ind]
	elif ref.ndim == 2 and gradient_category == 'read':
		return DWR[DERIV_L][i_ind,j_ind]
	elif gradient_category == 'read':
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
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	if ref.ndim == 2:
		y = -1e0*ref[i_ind,j_ind]
	else:
		k_ind = np.random.randint(ref.shape[2])
		y = -1e0*ref[i_ind,j_ind,k_ind]
	
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
