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
#DERIV_L = L1_UNDER
DERIV_L = L2_UNDER
#DERIV_L = F_UNDER
#DERIV_L = SHIFT
#DERIV_L = IN_GATE
#DERIV_L = KEY
#DERIV_L = BETA
#DERIV_L = ADD
#DERIV_L = ERASE
#DERIV_L = GAMMA
#gradient_category = 'write'
#gradient_category = 'read'
gradient_category = 'under'
####
if gradient_category == 'under':
	ref = WUNDER[DERIV_L]
elif gradient_category == 'read':
	ref = WR[DERIV_L]
else:
	ref = WW[DERIV_L]

########
def weight_address(W, B, O_PREV, inputs, mem_prev):
	O = [None]*len(O_PREV)
	
	# content
	O[KEY] = linear_2d_F(W[KEY], inputs)
	O[BETA] = linear_F(W[BETA], inputs) + B[BETA]
	O[CONTENT] = cosine_sim(O[KEY], mem_prev)
	O[CONTENT_FOCUSED] = focus_keys(O[CONTENT], O[BETA]) # beta*cos
	O[CONTENT_SM] = softmax(O[CONTENT_FOCUSED])
	
	# interpolate
	O[IN_GATE] = linear_F_sigmoid(W[IN_GATE], inputs)
	O[IN] = interpolate_softmax(O[IN_GATE], O[CONTENT_SM], O_PREV[F])
	
	# shift
	O[SHIFT] = linear_2d_F_softmax(W[SHIFT], inputs)
	O[SHIFTED] = shift_w(O[SHIFT], O[IN])
	
	# sharpen
	O[GAMMA] = relu(linear_F(W[GAMMA], inputs), thresh=1)
	O[SHARPENED] = sharpen(O[SHIFTED], O[GAMMA])
	
	O[F] = O[SHARPENED]
	
	return O

def forward_pass(WUNDER, WR,WW,BR,BW, OR_PREV, OW_PREV, mem_prev, x_cur):
	OUNDER = [None]*len(WUNDER)
	
	# processing underneath read/write heads
	OUNDER[L1_UNDER] = sq_F(WUNDER[L1_UNDER], x_cur)
	OUNDER[L2_UNDER] = sq_F(WUNDER[L2_UNDER], OUNDER[L1_UNDER])
	OUNDER[F_UNDER] = sq_F(WUNDER[F_UNDER], OUNDER[L2_UNDER])
	
	# read/write heads
	OR = weight_address(WR, BR, OR_PREV, OUNDER[F_UNDER], mem_prev)
	OW = weight_address(WW, BW, OW_PREV, OUNDER[F_UNDER], mem_prev)
	
	# erase/add output
	OW[ERASE] = linear_2d_F(WW[ERASE], OUNDER[F_UNDER])
	OW[ADD] = linear_2d_F(WW[ADD], OUNDER[F_UNDER])
	
	# read then write to mem
	read_mem = linear_F(OR[F], mem_prev)
	
	mem = mem_prev * (1 - add_mem(OW[F], OW[ERASE])) + add_mem(OW[F], OW[ADD])
	
	return OR,OW,mem,read_mem,OUNDER

# gradients for layers underneath the read/write heads
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


#### intermediate gradients used in several places in do_dw__inputs() and do_dw__mem_prev()
def do_do_content_focused__(O, do_do_in):
	do_in_do_content_sm = interpolate_softmax_do_content(O[IN], O[IN_GATE], O[CONTENT_SM])
	do_content_sm_do_content_focused = softmax_dlayer_in_nsum(O[CONTENT_SM])
	do_do_content_focused = mult_partials_chain((do_do_in, do_in_do_content_sm, do_content_sm_do_content_focused), (O[IN], O[CONTENT_SM]))
	
	return do_do_content_focused

# one step farther down the path from do_do_content_focused__()
def do_do_content__(O, do_do_in):
	do_do_content_focused = do_do_content_focused__(O, do_do_in)
	do_content_focused_do_content = focus_key_dkeys_nsum(O[CONTENT], O[BETA])
	do_do_content = mult_partials(do_do_content_focused, do_content_focused_do_content, O[CONTENT_FOCUSED])
	
	return do_do_content

########## ...
def do_dw__inputs(W, WUNDER, o_prev, OUNDER, DO_DWUNDER, O, DO_DW, DO_DB, mem_prev, x, do_do_in):
	DO_DW_NEW = copy.deepcopy(DO_DW); DO_DB_NEW = copy.deepcopy(DO_DB)
	DO_DWUNDER_NEW = copy.deepcopy(DO_DWUNDER)
	
	## sharpen weights
	do_dgammarelu = dsharpen_dgamma(O[SHIFTED], O[GAMMA])
	dgammarelu_dgamma = relu_dlayer_in(O[GAMMA], thresh=1)
	do_dgamma = mult_partials(do_dgammarelu, dgammarelu_dgamma, O[GAMMA])
	dgamma_dwgamma = linear_F_dF_nsum_g(W[GAMMA], OUNDER[F_UNDER])
	dgamma_dg3under = linear_F_dx_nsum_g(W[GAMMA], OUNDER[F_UNDER])
	DO_DW_NEW[GAMMA] += mult_partials(do_dgamma, dgamma_dwgamma, O[GAMMA])
	do_dg3under = np.squeeze(mult_partials(do_dgamma, dgamma_dg3under, O[GAMMA]))
	
	## shift weights
	do_dgshifted = dsharpen_dw(O[SHIFTED], O[GAMMA])
	dgshifted_dgshift = shift_w_dshift_out_nsum(O[IN])
	do_dgshift = mult_partials(do_dgshifted, dgshifted_dgshift, O[SHARPENED])
	dgshift_dwshift = linear_2d_F_softmax_dF_nsum(O[SHIFT], W[SHIFT], OUNDER[F_UNDER])
	dgshift_dg3under = linear_2d_F_softmax_dx_nsum(O[SHIFT], W[SHIFT])
	DO_DW_NEW[SHIFT] += mult_partials(do_dgshift, dgshift_dwshift, O[SHIFT])
	do_dg3under += mult_partials(do_dgshift, dgshift_dg3under, O[SHIFT])
	
	## interp. gradients (wrt gin_gate)
	do_in_dgin_gate = interpolate_softmax_dinterp_gate_out(O[IN], O[IN_GATE], O[CONTENT_SM], o_prev)
	do_dgin_gate = mult_partials(do_do_in, do_in_dgin_gate, O[IN])
	dgin_gate_dwin = linear_F_sigmoid_dF_nsum_g(O[IN_GATE], W[IN_GATE], OUNDER[F_UNDER])
	dgin_gate_dg3under = linear_F_sigmoid_dx_nsum_g(O[IN_GATE], W[IN_GATE], OUNDER[F_UNDER])
	DO_DW_NEW[IN_GATE] += mult_partials(do_dgin_gate, dgin_gate_dwin, O[IN_GATE])
	do_dg3under += np.squeeze(mult_partials(do_dgin_gate, dgin_gate_dg3under, O[IN_GATE]))
	
	## interp. gradients (wrt o_content; key)
	do_do_content = do_do_content__(O, do_do_in)
	do_content_dgkey = cosine_sim_expand_dkeys(O[KEY], mem_prev)
	do_dgkey = mult_partials(do_do_content, do_content_dgkey, O[CONTENT])
	dgkey_dwkey = linear_2d_F_dF_nsum(W[KEY], OUNDER[F_UNDER])
	dgkey_dg3under = linear_2d_F_dx_nsum(W[KEY])
	DO_DW_NEW[KEY] += mult_partials(do_dgkey, dgkey_dwkey, O[KEY])
	do_dg3under += mult_partials(do_dgkey, dgkey_dg3under, O[KEY])
	
	## interp. gradients (wrt beta)
	do_do_content_focused = do_do_content_focused__(O, do_do_in)
	do_content_focused_dgbeta = focus_key_dbeta_out_nsum(O[CONTENT], O[BETA])
	do_dgbeta = mult_partials(do_do_content_focused, do_content_focused_dgbeta, O[CONTENT_FOCUSED])
	DO_DB_NEW[BETA] += do_dgbeta
	dgbeta_dwbeta = linear_F_dF_nsum_g(W[BETA], OUNDER[F_UNDER])
	dgbeta_dg3under = linear_F_dx_nsum_g(W[BETA], OUNDER[F_UNDER])
	DO_DW_NEW[BETA] += mult_partials(do_dgbeta, dgbeta_dwbeta, O[BETA])
	do_dg3under += np.squeeze(mult_partials(do_dgbeta, dgbeta_dg3under, O[BETA]))
	
	## combine weights under gradients
	DG3UNDER_DW = dunder_dw(WUNDER, OUNDER, x)
	DO_DWUNDER_NEW = mult_partials__layers(do_dg3under, DG3UNDER_DW, np.squeeze(OUNDER[F_UNDER]), DO_DWUNDER_NEW)
	
	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW

########## ...
def do_dw__o_prev(W, o_prev, DO_DW, DO_DB, DO_DWUNDER, O, do_do_in):
	do_in_do_prev = interpolate_softmax_do_prev(O[IN], O[IN_GATE], o_prev)
	do_do_prev = mult_partials(do_do_in, do_in_do_prev, O[IN])
	
	DO_DW_NEW = mult_partials__layers(do_do_prev, DO_DW, o_prev)
	DO_DB_NEW = mult_partials__layers(do_do_prev, DO_DB, o_prev)
	DO_DWUNDER_NEW = mult_partials__layers(do_do_prev, DO_DWUNDER, o_prev)
	
	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW

def do_dw__mem_prev(W, DO_DW, DO_DB, DO_DWUNDER, O, mem_prev, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, do_do_in):
	do_do_content = do_do_content__(O, do_do_in)
	do_content_dmem_prev = cosine_sim_expand_dmem(O[KEY], mem_prev)
	do_dmem_prev = mult_partials(do_do_content, do_content_dmem_prev, O[CONTENT])
	
	DO_DW_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWW, mem_prev, DO_DW)
	DO_DB_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DBW, mem_prev, DO_DB)
	DO_DWUNDER_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWUNDER, mem_prev, DO_DWUNDER)
	
	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW

def mem_partials(DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DOW_DWW, DOW_DBW,DOW_DWUNDER, OW_PREV, OUNDER_PREV, WW,BW, WUNDER, x_prev, mem_prev_prev):
	DG3UNDER_DW = dunder_dw(WUNDER, OUNDER_PREV, x_prev)
	# mem = mem_prev*(1 - e) + a
	# dmem = dmem_prev*(1 - e) - mem_prev*de + da
	
	# write gradients (erase)
	e = add_mem(OW_PREV[F], OW_PREV[ERASE])
	
	mem_prev_de_dow = -add_mem_dgw(OW_PREV[ERASE]) * mem_prev_prev[:,:,np.newaxis,np.newaxis] # -mem_prev * de
	
	# dmem_prev * (1 - e)
	DMEM_PREV_DWW_NEW = pointwise_mult_partials__layers(1 - e, DMEM_PREV_DWW)
	DMEM_PREV_DBW_NEW = pointwise_mult_partials__layers(1 - e, DMEM_PREV_DBW)
	DMEM_PREV_DWUNDER_NEW = pointwise_mult_partials__layers(1 - e, DMEM_PREV_DWUNDER)
	
	# dmem_prev * (1 - e) - mem_prev * de
	DMEM_PREV_DWW_NEW = mult_partials__layers(mem_prev_de_dow, DOW_DWW, OW_PREV[F], DMEM_PREV_DWW_NEW)
	DMEM_PREV_DBW_NEW = mult_partials__layers(mem_prev_de_dow, DOW_DBW, OW_PREV[F], DMEM_PREV_DBW_NEW)
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(mem_prev_de_dow, DOW_DWUNDER, OW_PREV[F], DMEM_PREV_DWUNDER_NEW)
	
	###
	# W[ERASE] gradients (de wrt W[ERASE])
	mem_prev_de_derase_out = -add_mem_dadd_out(OW_PREV[F]) * mem_prev_prev[:,:,np.newaxis,np.newaxis]
	
	derase_out_dwadd = linear_2d_F_dF_nsum(WW[ERASE], OUNDER_PREV[F_UNDER])
	DMEM_PREV_DWW_NEW[ERASE] += mult_partials(mem_prev_de_derase_out, derase_out_dwadd, OW_PREV[ERASE]) # de_dwadd
	
	# under: (wrt inputs)
	derase_out_dg3under = linear_2d_F_dx_nsum(WW[ERASE])
	mem_prev_de_dg3under = mult_partials(mem_prev_de_derase_out, derase_out_dg3under, OW_PREV[ERASE])
	
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(mem_prev_de_dg3under, DG3UNDER_DW, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DWUNDER_NEW)
	
	
	################
	# write gradients (add)
	da_dow = add_mem_dgw(OW_PREV[ADD]) # da
	
	DMEM_PREV_DWW_NEW = mult_partials__layers(da_dow, DOW_DWW, OW_PREV[F], DMEM_PREV_DWW_NEW) # da_dlayer
	DMEM_PREV_DBW_NEW = mult_partials__layers(da_dow, DOW_DBW, OW_PREV[F], DMEM_PREV_DBW_NEW) # da_dlayer
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(da_dow, DOW_DWUNDER, OW_PREV[F], DMEM_PREV_DWUNDER_NEW)
	
	###
	# W[ADD] gradients
	da_dadd_out = add_mem_dadd_out(OW_PREV[F])
	
	dadd_out_dwadd = linear_2d_F_dF_nsum(WW[ADD], OUNDER_PREV[F_UNDER])
	DMEM_PREV_DWW_NEW[ADD] += mult_partials(da_dadd_out, dadd_out_dwadd, OW_PREV[ADD]) # da_dwadd
	
	# under: (wrt inputs)
	dadd_out_dg3under = linear_2d_F_dx_nsum(WW[ADD])
	da_dg3under = mult_partials(da_dadd_out, dadd_out_dg3under, OW_PREV[ADD]) # da_dwunder
	
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(da_dg3under, DG3UNDER_DW, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DWUNDER_NEW)
	
	return DMEM_PREV_DWW_NEW, DMEM_PREV_DBW_NEW, DMEM_PREV_DWUNDER_NEW

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
		OR_PREV, OW_PREV, mem_prev, read_mem = forward_pass(WUNDER, WR,WW,BR,BW, OR_PREV, OW_PREV, mem_prev, x[frame])[:4]
	
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
	DOR_DBR = copy.deepcopy(DOR_DBRi); DOW_DBW = copy.deepcopy(DOW_DBWi); DOR_DBW = copy.deepcopy(DOR_DBWi)
	DOW_DWUNDER = copy.deepcopy(DOW_DWUNDERi); DOR_DWUNDER = copy.deepcopy(DOR_DWUNDERi)
	mem_prev = copy.deepcopy(mem_previ); mem_prev_prev = copy.deepcopy(mem_previ)
	DMEM_PREV_DWW = copy.deepcopy(DMEM_PREV_DWWi); DMEM_PREV_DWUNDER = copy.deepcopy(DMEM_PREV_DWUNDERi)
	DMEM_PREV_DBW = copy.deepcopy(DMEM_PREV_DBWi)
	
	###
	for frame in range(1,N_FRAMES+1):
		# forward
		OR, OW, mem, read_mem, OUNDER = forward_pass(WUNDER, WR,WW,BR,BW, OR_PREV, OW_PREV, mem_prev, x[frame])
		
		# reverse
		dor_dgsharpen = dsharpen_dw(OR[SHIFTED], OR[GAMMA])
		dow_prev_dgsharpen = dsharpen_dw(OW_PREV[SHIFTED], OW_PREV[GAMMA])
		
		dgsharpen_dor_in = shift_w_dw_interp_nsum(OR[SHIFT])
		dgsharpen_dow_prev_in = shift_w_dw_interp_nsum(OW_PREV[SHIFT])
		
		dor_dor_in = mult_partials(dor_dgsharpen, dgsharpen_dor_in, OR[SHARPENED])
		dow_prev_dow_prev_in = mult_partials(dow_prev_dgsharpen, dgsharpen_dow_prev_in, OW_PREV[SHARPENED])
		
		# partials for write head output (OW)
		if frame > 1:
			DOW_DWW, DOW_DBW, DOW_DWUNDER = do_dw__o_prev(WW, OW_PREV_PREV[F], DOW_DWW, DOW_DBW, DOW_DWUNDER, OW_PREV, dow_prev_dow_prev_in)
			DOW_DWW, DOW_DBW, DOW_DWUNDER = do_dw__mem_prev(WW, DOW_DWW, DOW_DBW, DOW_DWUNDER, OW_PREV, \
												mem_prev_prev, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, dow_prev_dow_prev_in)
			DOW_DWW, DOW_DBW, DOW_DWUNDER = do_dw__inputs(WW, WUNDER, OW_PREV_PREV[F], OUNDER_PREV, DOW_DWUNDER, \
												OW_PREV, DOW_DWW, DOW_DBW, mem_prev_prev, x[frame-1], dow_prev_dow_prev_in)
			
			DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER = mem_partials(DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DOW_DWW, \
												DOW_DBW, DOW_DWUNDER, OW_PREV, OUNDER_PREV, WW, BW, WUNDER, x[frame-1], mem_prev_prev)
		
		# partials from read head output (OR)
		DOR_DWR, DOR_DBR, DOR_DWUNDER = do_dw__o_prev(WR, OR_PREV[F], DOR_DWR, DOR_DBR, DOR_DWUNDER, OR, dor_dor_in)
		DOR_DWR, DOR_DBR, DOR_DWUNDER = do_dw__inputs(WR, WUNDER, OR_PREV[F], OUNDER, DOR_DWUNDER, OR, DOR_DWR, DOR_DBR, mem_prev, x[frame], dor_dor_in)
		
		DOR_DWW, DOR_DBW = do_dw__o_prev(WR, OR_PREV[F], DOR_DWW, DOR_DBW, DOR_DWUNDER, OR, dor_dor_in)[:2] #?
		DOR_DWW, DOR_DBW, DOR_DWUNDER = do_dw__mem_prev(WR, DOR_DWW, DOR_DBW, DOR_DWUNDER, OR, mem_prev, DMEM_PREV_DWW, \
											DMEM_PREV_DBW, DMEM_PREV_DWUNDER, dor_dor_in)
	
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
