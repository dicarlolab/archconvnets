import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

def weight_address(W, B, O_PREV, inputs, mem_prev):
	O = [None]*len(O_PREV)
	
	# content
	O[KEY] = linear_2d_F(W[KEY], inputs) + B[KEY]
	O[BETA] = linear_F(W[BETA], inputs) + B[BETA]
	O[CONTENT] = cosine_sim(O[KEY], mem_prev)
	O[CONTENT_FOCUSED] = focus_keys(O[CONTENT], O[BETA]) # beta*cos
	O[CONTENT_SM] = softmax(O[CONTENT_FOCUSED])
	
	# interpolate
	O[IN_GATE] = sigmoid(linear_F(W[IN_GATE], inputs) + B[IN_GATE])
	O[IN] = interpolate_softmax(O[IN_GATE], O[CONTENT_SM], O_PREV[F])
	
	# shift
	O[SHIFT] = softmax(linear_2d_F(W[SHIFT], inputs) + B[SHIFT])
	O[SHIFTED] = shift_w(O[SHIFT], O[IN])
	
	# sharpen
	O[GAMMA] = relu(linear_F(W[GAMMA], inputs) + B[GAMMA], thresh=1)
	O[SHARPENED] = sharpen(O[SHIFTED], O[GAMMA])
	
	O[F] = O[SHARPENED]
	
	return O

def forward_pass(WUNDER,BUNDER, WR,WW,BR,BW, WABOVE, BABOVE, OR_PREV, OW_PREV, mem_prev, x_cur):
	OUNDER = [None]*len(WUNDER)
	OABOVE = [None]*len(WABOVE)
	
	# processing underneath read/write heads
	OUNDER[L1_UNDER] = relu(linear_F(WUNDER[L1_UNDER], x_cur) + BUNDER[L1_UNDER])
	OUNDER[L2_UNDER] = sq_F(WUNDER[L2_UNDER], OUNDER[L1_UNDER]) + BUNDER[L2_UNDER]
	OUNDER[F_UNDER] = sq_F(WUNDER[F_UNDER], OUNDER[L2_UNDER]) #+ BUNDER[F_UNDER]
	
	# read/write heads
	OR = weight_address(WR, BR, OR_PREV, OUNDER[F_UNDER], mem_prev)
	OW = weight_address(WW, BW, OW_PREV, OUNDER[F_UNDER], mem_prev)
	
	# erase/add output
	OW[ERASE] = sigmoid(linear_2d_F(WW[ERASE], OUNDER[F_UNDER]) + BW[ERASE])
	OW[ADD] = linear_2d_F(WW[ADD], OUNDER[F_UNDER]) + BW[ADD]
	
	# read then write to mem
	read_mem = linear_F(OR[F], mem_prev)
	
	mem = mem_prev * (1 - add_mem(OW[F], OW[ERASE])) + add_mem(OW[F], OW[ADD])

	# above
	OABOVE[L1_ABOVE] = relu(linear_F(WABOVE[L1_ABOVE], read_mem.reshape(C*mem_length,1))) + BABOVE[L1_ABOVE]
	OABOVE[F_ABOVE] = linear_F(WABOVE[F_ABOVE], OABOVE[L1_ABOVE]) + BABOVE[F_ABOVE]
	
		
	return OR,OW,mem,read_mem,OUNDER

# gradients for layers underneath the read/write heads
# (used in mem_partials() and do_dw__inputs()
def dunder(WUNDER, BUNDER, OUNDER, x):
	DG3UNDER_DW = [None] * len(OUNDER); DG3UNDER_DB = [None] * len(OUNDER)
	
	DG3UNDER_DB[F_UNDER] = np.ones((9,9,1)) ###########np.ones_like(OUNDER[F_UNDER])
	DG3UNDER_DW[F_UNDER] = sq_dF_nsum(WUNDER[F_UNDER], OUNDER[L2_UNDER], OUNDER[F_UNDER])
	
	dg3under_dg2under = sq_dlayer_in_nsum(WUNDER[F_UNDER], OUNDER[L2_UNDER])
	dg2under_dw2under = sq_dF_nsum(WUNDER[L2_UNDER], OUNDER[L1_UNDER], OUNDER[L2_UNDER])
	DG3UNDER_DB[L2_UNDER] = dg3under_dg2under[:,:,np.newaxis]
	DG3UNDER_DW[L2_UNDER] = mult_partials(dg3under_dg2under, dg2under_dw2under,  np.squeeze(OUNDER[L2_UNDER]))
	
	dg2under_dg1under_relu = sq_dlayer_in_nsum(WUNDER[L2_UNDER], OUNDER[L1_UNDER])
	dg1under_relu_dg1under = relu_dlayer_in(OUNDER[L1_UNDER])
	dg2under_dg1under = np.squeeze(mult_partials(dg2under_dg1under_relu[:,:,np.newaxis], dg1under_relu_dg1under, OUNDER[L1_UNDER]))
	dg1under_dw1under = linear_F_dF_nsum_g(WUNDER[L1_UNDER], x);    dg1under_dw1under = np.squeeze(dg1under_dw1under)
	dg2under_dw1under = mult_partials(dg2under_dg1under, dg1under_dw1under,  np.squeeze(OUNDER[L1_UNDER]))
	DG3UNDER_DB[L1_UNDER] = mult_partials(dg3under_dg2under, dg2under_dg1under, np.squeeze(OUNDER[L2_UNDER]))[:,:,np.newaxis]
	DG3UNDER_DW[L1_UNDER] = mult_partials(dg3under_dg2under, dg2under_dw1under, np.squeeze(OUNDER[L2_UNDER]))
	
	return DG3UNDER_DW, DG3UNDER_DB


#### intermediate gradients used in several places in do_dw__inputs() and do_dw__mem_prev()
def do_do_content_focused__(O, do_do_in):
	do_in_do_content_sm = interpolate_softmax_do_content(O[IN], O[IN_GATE], O[CONTENT_SM])
	do_content_sm_do_content_focused = softmax_dlayer_in_nsum(O[CONTENT_SM])
	do_do_content_focused = mult_partials_chain((do_do_in, do_in_do_content_sm, do_content_sm_do_content_focused), (O[IN], O[CONTENT_SM]))
	
	return do_do_content_focused

# one step farther down the path from do_do_content_focused__()
# used in do_dw__inputs() and do_dw__mem_prev()
def do_do_content__(O, do_do_in):
	do_do_content_focused = do_do_content_focused__(O, do_do_in)
	do_content_focused_do_content = focus_key_dkeys_nsum(O[CONTENT], O[BETA])
	do_do_content = mult_partials(do_do_content_focused, do_content_focused_do_content, O[CONTENT_FOCUSED])
	
	return do_do_content

########## ...
# 25.2% of reverse_pass_partials()
#@profile
def do_dw__inputs(W, WUNDER, BUNDER, o_prev, OUNDER, DO_DWUNDER, DO_DBUNDER, O, DO_DW, DO_DB, mem_prev, x, do_do_in):
	DO_DW_NEW = copy.deepcopy(DO_DW); DO_DB_NEW = copy.deepcopy(DO_DB) # 3.5%
	DO_DWUNDER_NEW = copy.deepcopy(DO_DWUNDER); DO_DBUNDER_NEW = copy.deepcopy(DO_DBUNDER)
	
	## sharpen weights
	do_dgammarelu = dsharpen_dgamma(O[SHIFTED], O[GAMMA])
	dgammarelu_dgamma = relu_dlayer_in(O[GAMMA], thresh=1)
	do_dgamma = mult_partials(do_dgammarelu, dgammarelu_dgamma, O[GAMMA])
	DO_DB_NEW[GAMMA] += do_dgamma
	dgamma_dwgamma = linear_F_dF_nsum_g(W[GAMMA], OUNDER[F_UNDER])
	dgamma_dg3under = linear_F_dx_nsum_g(W[GAMMA], OUNDER[F_UNDER])
	DO_DW_NEW[GAMMA] += mult_partials(do_dgamma, dgamma_dwgamma, O[GAMMA])
	do_dg3under = np.squeeze(mult_partials(do_dgamma, dgamma_dg3under, O[GAMMA]))
	
	## shift weights
	do_dgshiftedsm = dsharpen_dw(O[SHIFTED], O[GAMMA])
	dgshiftedsm_dgshiftsm = shift_w_dshift_out_nsum(O[IN])
	do_dgshiftsm = mult_partials(do_dgshiftedsm, dgshiftedsm_dgshiftsm, O[SHARPENED])
	dgshiftsm_gshift = softmax_dlayer_in_nsum(O[SHIFT])
	do_dgshift = mult_partials(do_dgshiftsm, dgshiftsm_gshift, O[SHIFT])
	DO_DB_NEW[SHIFT] += do_dgshift
	dgshift_dwshift = linear_2d_F_dF_nsum(W[SHIFT], OUNDER[F_UNDER])
	dgshift_dg3under = linear_2d_F_dx_nsum(W[SHIFT])
	DO_DW_NEW[SHIFT] += mult_partials(do_dgshift, dgshift_dwshift, O[SHIFT])
	do_dg3under += mult_partials(do_dgshift, dgshift_dg3under, O[SHIFT])
	
	## interp. gradients (wrt gin_gate)
	do_in_dgin_gate_sig = interpolate_softmax_dinterp_gate_out(O[IN], O[IN_GATE], O[CONTENT_SM], o_prev) # 4.2%
	do_dgin_gate_sig = mult_partials(do_do_in, do_in_dgin_gate_sig, O[IN])
	dgin_gate_sig_dgin_gate = sigmoid_dlayer_in(O[IN_GATE])
	do_dgin_gate = mult_partials(do_dgin_gate_sig, dgin_gate_sig_dgin_gate, O[IN_GATE])
	DO_DB_NEW[IN_GATE] += do_dgin_gate
	dgin_gate_dwin = linear_F_dF_nsum_g(W[IN_GATE], OUNDER[F_UNDER])
	dgin_gate_dg3under = linear_F_dx_nsum_g(W[IN_GATE], OUNDER[F_UNDER])
	DO_DW_NEW[IN_GATE] += mult_partials(do_dgin_gate, dgin_gate_dwin, O[IN_GATE])
	do_dg3under += np.squeeze(mult_partials(do_dgin_gate, dgin_gate_dg3under, O[IN_GATE]))
	
	## interp. gradients (wrt o_content; key)
	do_do_content = do_do_content__(O, do_do_in) # 14%
	do_content_dgkey = cosine_sim_expand_dkeys(O[KEY], mem_prev) # 12.3%
	do_dgkey = mult_partials(do_do_content, do_content_dgkey, O[CONTENT])
	DO_DB_NEW[KEY] += do_dgkey
	dgkey_dwkey = linear_2d_F_dF_nsum(W[KEY], OUNDER[F_UNDER])
	dgkey_dg3under = linear_2d_F_dx_nsum(W[KEY])
	DO_DW_NEW[KEY] += mult_partials(do_dgkey, dgkey_dwkey, O[KEY]) # 28.6%
	do_dg3under += mult_partials(do_dgkey, dgkey_dg3under, O[KEY])
	
	## interp. gradients (wrt beta)
	do_do_content_focused = do_do_content_focused__(O, do_do_in) # 12.2%
	do_content_focused_dgbeta = focus_key_dbeta_out_nsum(O[CONTENT], O[BETA])
	do_dgbeta = mult_partials(do_do_content_focused, do_content_focused_dgbeta, O[CONTENT_FOCUSED])
	DO_DB_NEW[BETA] += do_dgbeta
	dgbeta_dwbeta = linear_F_dF_nsum_g(W[BETA], OUNDER[F_UNDER])
	dgbeta_dg3under = linear_F_dx_nsum_g(W[BETA], OUNDER[F_UNDER])
	DO_DW_NEW[BETA] += mult_partials(do_dgbeta, dgbeta_dwbeta, O[BETA])
	do_dg3under += np.squeeze(mult_partials(do_dgbeta, dgbeta_dg3under, O[BETA]))
	
	## combine weights under gradients
	DG3UNDER_DW, DG3UNDER_DB = dunder(WUNDER, BUNDER, OUNDER, x)
	DO_DWUNDER_NEW = mult_partials__layers(do_dg3under, DG3UNDER_DW, np.squeeze(OUNDER[F_UNDER]), DO_DWUNDER_NEW)
	DO_DBUNDER_NEW = mult_partials__layers(do_dg3under, DG3UNDER_DB, np.squeeze(OUNDER[F_UNDER]), DO_DBUNDER_NEW)
	
	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW, DO_DBUNDER_NEW

########## ...
# 34.2% of reverse_pass_partials()
def do_dw__o_prev(W, o_prev, DO_DW, DO_DB, DO_DWUNDER,DO_DBUNDER, O, do_do_in):
	do_in_do_prev = interpolate_softmax_do_prev(O[IN], O[IN_GATE], o_prev) # 6.2%
	do_do_prev = mult_partials(do_do_in, do_in_do_prev, O[IN])
	
	DO_DW_NEW = mult_partials__layers(do_do_prev, DO_DW, o_prev) # 67.9%
	DO_DB_NEW = mult_partials__layers(do_do_prev, DO_DB, o_prev) # 8.8%
	DO_DWUNDER_NEW = mult_partials__layers(do_do_prev, DO_DWUNDER, o_prev) # 13.3%
	DO_DBUNDER_NEW = mult_partials__layers(do_do_prev, DO_DBUNDER, o_prev)
	
	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW, DO_DBUNDER_NEW

#########
# 20.8% of reverse_pass_partials()
def do_dw__mem_prev(W, DO_DW, DO_DB, DO_DWUNDER,DO_DBUNDER, O, mem_prev, DMEM_PREV_DWW, DMEM_PREV_DBW, \
			DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, do_do_in):
	do_do_content = do_do_content__(O, do_do_in) # 17.9%
	do_content_dmem_prev = cosine_sim_expand_dmem(O[KEY], mem_prev) # 15.5%
	do_dmem_prev = mult_partials(do_do_content, do_content_dmem_prev, O[CONTENT])
	
	DO_DW_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWW, mem_prev, DO_DW) # 49.6%
	DO_DB_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DBW, mem_prev, DO_DB)
	DO_DWUNDER_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWUNDER, mem_prev, DO_DWUNDER)
	DO_DBUNDER_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DBUNDER, mem_prev, DO_DBUNDER)
	
	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW, DO_DBUNDER_NEW

# 18.6% of reverse_pass_partials()
def mem_partials(DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER,DMEM_PREV_DBUNDER, DOW_DWW, DOW_DBW,DOW_DWUNDER,DOW_DBUNDER,\
			OW_PREV, OUNDER_PREV, WW,BW, WUNDER, BUNDER, x_prev, mem_prev_prev):
	DG3UNDER_DW, DG3UNDER_DB = dunder(WUNDER, BUNDER, OUNDER_PREV, x_prev)
	# mem = mem_prev*(1 - e) + a
	# dmem = dmem_prev*(1 - e) - mem_prev*de + da
	
	# write gradients (erase)
	e = add_mem(OW_PREV[F], OW_PREV[ERASE])
	
	mem_prev_de_dow = -add_mem_dgw(OW_PREV[ERASE]) * mem_prev_prev[:,:,np.newaxis,np.newaxis] # -mem_prev * de
	
	# dmem_prev * (1 - e)
	DMEM_PREV_DWW_NEW = pointwise_mult_partials__layers(1 - e, DMEM_PREV_DWW)
	DMEM_PREV_DBW_NEW = pointwise_mult_partials__layers(1 - e, DMEM_PREV_DBW)
	DMEM_PREV_DWUNDER_NEW = pointwise_mult_partials__layers(1 - e, DMEM_PREV_DWUNDER)
	DMEM_PREV_DBUNDER_NEW = pointwise_mult_partials__layers(1 - e, DMEM_PREV_DBUNDER)
	
	# dmem_prev * (1 - e) - mem_prev * de
	DMEM_PREV_DWW_NEW = mult_partials__layers(mem_prev_de_dow, DOW_DWW, OW_PREV[F], DMEM_PREV_DWW_NEW)
	DMEM_PREV_DBW_NEW = mult_partials__layers(mem_prev_de_dow, DOW_DBW, OW_PREV[F], DMEM_PREV_DBW_NEW)
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(mem_prev_de_dow, DOW_DWUNDER, OW_PREV[F], DMEM_PREV_DWUNDER_NEW)
	DMEM_PREV_DBUNDER_NEW = mult_partials__layers(mem_prev_de_dow, DOW_DBUNDER, OW_PREV[F], DMEM_PREV_DBUNDER_NEW)
	
	###
	# W[ERASE] gradients (de wrt W[ERASE])
	mem_prev_de_derase_out_sig = -add_mem_dadd_out(OW_PREV[F]) * mem_prev_prev[:,:,np.newaxis,np.newaxis]
	derase_out_sig_derase_out = sigmoid_dlayer_in(OW_PREV[ERASE])
	mem_prev_de_derase_out = mult_partials(mem_prev_de_derase_out_sig, derase_out_sig_derase_out, OW_PREV[ERASE])
	DMEM_PREV_DBW_NEW[ERASE] += mem_prev_de_derase_out
	derase_out_dwadd = linear_2d_F_dF_nsum(WW[ERASE], OUNDER_PREV[F_UNDER])
	DMEM_PREV_DWW_NEW[ERASE] += mult_partials(mem_prev_de_derase_out, derase_out_dwadd, OW_PREV[ERASE]) # de_dwadd
	
	# under: (wrt inputs)
	derase_out_dg3under = linear_2d_F_dx_nsum(WW[ERASE])
	mem_prev_de_dg3under = mult_partials(mem_prev_de_derase_out, derase_out_dg3under, OW_PREV[ERASE])
	
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(mem_prev_de_dg3under, DG3UNDER_DW, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DWUNDER_NEW)
	DMEM_PREV_DBUNDER_NEW = mult_partials__layers(mem_prev_de_dg3under, DG3UNDER_DB, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DBUNDER_NEW)
	
	
	################
	# write gradients (add)
	da_dow = add_mem_dgw(OW_PREV[ADD]) # da
	
	DMEM_PREV_DWW_NEW = mult_partials__layers(da_dow, DOW_DWW, OW_PREV[F], DMEM_PREV_DWW_NEW) # da_dlayer
	DMEM_PREV_DBW_NEW = mult_partials__layers(da_dow, DOW_DBW, OW_PREV[F], DMEM_PREV_DBW_NEW) # da_dlayer
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(da_dow, DOW_DWUNDER, OW_PREV[F], DMEM_PREV_DWUNDER_NEW)
	DMEM_PREV_DBUNDER_NEW = mult_partials__layers(da_dow, DOW_DBUNDER, OW_PREV[F], DMEM_PREV_DBUNDER_NEW)
	
	###
	# W[ADD] gradients
	da_dadd_out = add_mem_dadd_out(OW_PREV[F])
	DMEM_PREV_DBW_NEW[ADD] += da_dadd_out
	dadd_out_dwadd = linear_2d_F_dF_nsum(WW[ADD], OUNDER_PREV[F_UNDER])
	DMEM_PREV_DWW_NEW[ADD] += mult_partials(da_dadd_out, dadd_out_dwadd, OW_PREV[ADD]) # da_dwadd
	
	# under: (wrt inputs)
	dadd_out_dg3under = linear_2d_F_dx_nsum(WW[ADD])
	da_dg3under = mult_partials(da_dadd_out, dadd_out_dg3under, OW_PREV[ADD]) # da_dwunder
	
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(da_dg3under, DG3UNDER_DW, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DWUNDER_NEW)
	DMEM_PREV_DBUNDER_NEW = mult_partials__layers(da_dg3under, DG3UNDER_DB, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DBUNDER_NEW)
	
	return DMEM_PREV_DWW_NEW, DMEM_PREV_DBW_NEW, DMEM_PREV_DWUNDER_NEW, DMEM_PREV_DBUNDER_NEW
	

### compute state partials (stores history, in a sense) 
# 74.7% of main()
def reverse_pass_partials(WUNDER,BUNDER, WR,WW,BR,BW, OUNDER, OUNDER_PREV, OR, OR_PREV, OW_PREV, \
				OW_PREV_PREV, mem_prev, mem_prev_prev, x, x_prev, frame, DOW_DWW, DOW_DBW, \
				DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, \
				DMEM_PREV_DBUNDER, DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW):
	dor_dgsharpen = dsharpen_dw(OR[SHIFTED], OR[GAMMA])
	dow_prev_dgsharpen = dsharpen_dw(OW_PREV[SHIFTED], OW_PREV[GAMMA])
	
	dgsharpen_dor_in = shift_w_dw_interp_nsum(OR[SHIFT])
	dgsharpen_dow_prev_in = shift_w_dw_interp_nsum(OW_PREV[SHIFT])
	
	dor_dor_in = mult_partials(dor_dgsharpen, dgsharpen_dor_in, OR[SHARPENED])
	dow_prev_dow_prev_in = mult_partials(dow_prev_dgsharpen, dgsharpen_dow_prev_in, OW_PREV[SHARPENED])
	
	# partials for write head output (OW)
	if frame > 1:
		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__o_prev(WW, OW_PREV_PREV[F], DOW_DWW, DOW_DBW, DOW_DWUNDER,\
											DOW_DBUNDER, OW_PREV, dow_prev_dow_prev_in) # 13.4%
		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__mem_prev(WW, DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, \
											mem_prev_prev, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, \
											DMEM_PREV_DBUNDER,  dow_prev_dow_prev_in) # 10.3%
		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__inputs(WW, WUNDER, BUNDER, OW_PREV_PREV[F], OUNDER_PREV, DOW_DWUNDER, \
											DOW_DBUNDER, OW_PREV, DOW_DWW, DOW_DBW, mem_prev_prev, x_prev, dow_prev_dow_prev_in) # 12.6%
		
		DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER = mem_partials(DMEM_PREV_DWW, DMEM_PREV_DBW, \
											DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, DOW_DWW, \
											DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, OUNDER_PREV, WW, BW, \
											WUNDER, BUNDER, x_prev, mem_prev_prev) # 18.6%
	
	# partials from read head output (OR)
	DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER = do_dw__o_prev(WR, OR_PREV[F], DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, OR, dor_dor_in) # 7.8%
	DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER = do_dw__inputs(WR, WUNDER, BUNDER, OR_PREV[F], OUNDER, DOR_DWUNDER, DOR_DBUNDER, \
										OR, DOR_DWR, DOR_DBR, mem_prev, x, dor_dor_in) # 12.3%
	
	DOR_DWW, DOR_DBW = do_dw__o_prev(WR, OR_PREV[F], DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, dor_dor_in)[:2] #?... 13.4%
	DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER = do_dw__mem_prev(WR, DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, \
										mem_prev, DMEM_PREV_DWW, \
										DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, dor_dor_in) # 10.4%
	
	return DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, \
			DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW



### compute full gradients from state partials
# 24.8 of main()
def full_gradients(read_mem, t, mem_prev, DOR_DWR, DOR_DBR, DOR_DWW, DOR_DBW, DOR_DWUNDER,DOR_DBUNDER, OR, DMEM_PREV_DWW, \
			DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER):
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	# read weights
	dread_mem_dor = linear_F_dF_nsum(mem_prev)
	derr_dor = mult_partials(derr_dread_mem, dread_mem_dor, read_mem)
	
	DWR = mult_partials_collapse__layers(derr_dor, DOR_DWR, OR[F]) # 18.3%
	DBR = mult_partials_collapse__layers(derr_dor, DOR_DBR, OR[F])
	DWW = mult_partials_collapse__layers(derr_dor, DOR_DWW, OR[F]) # 38.5%
	DBW = mult_partials_collapse__layers(derr_dor, DOR_DBW, OR[F])
	DWUNDER = mult_partials_collapse__layers(derr_dor, DOR_DWUNDER, OR[F])
	DBUNDER = mult_partials_collapse__layers(derr_dor, DOR_DBUNDER, OR[F])
	
	# write weights
	dread_mem_dmem_prev = linear_F_dx_nsum(OR[F])
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	DWW = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DWW, mem_prev, DWW) # 20%
	DBW = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DBW, mem_prev, DBW)
	DWUNDER = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DWUNDER, mem_prev, DWUNDER)
	DBUNDER = mult_partials_collapse__layers(derr_dmem_prev, DMEM_PREV_DBUNDER, mem_prev, DBUNDER)
	
	return DWR, DBR, DWW, DBW, DWUNDER, DBUNDER

