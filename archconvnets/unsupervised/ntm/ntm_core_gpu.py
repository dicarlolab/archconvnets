import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *
import archconvnets.unsupervised.ntm_module.ntm_module as nm
from archconvnets.unsupervised.ntm_module.ntm_module import init_buffer, set_list_buffer, return_list_buffer

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
	OUNDER[L2_UNDER] = relu(linear_F(WUNDER[L2_UNDER], OUNDER[L1_UNDER]) + BUNDER[L2_UNDER])
	OUNDER[F_UNDER] = relu(linear_F(WUNDER[F_UNDER], OUNDER[L2_UNDER]) + BUNDER[F_UNDER])
	
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
	OABOVE[L1_ABOVE] = relu(linear_F(WABOVE[L1_ABOVE], read_mem.reshape(C*mem_length,1)) + BABOVE[L1_ABOVE])
	OABOVE[F_ABOVE] = relu(linear_F(WABOVE[F_ABOVE], OABOVE[L1_ABOVE]) + BABOVE[F_ABOVE])
	
	return OR,OW,mem,read_mem,OUNDER,OABOVE

# gradients for layers underneath the read/write heads
# (used in mem_partials() and do_dw__inputs()
def dunder(WUNDER, BUNDER, OUNDER, x):
	DG3UNDER_DW = [None] * len(OUNDER); DG3UNDER_DB = [None] * len(OUNDER)

	dg3under_relu_dg3under = relu_dlayer_in(OUNDER[F_UNDER]).squeeze()
	dg3under_dw3under = linear_F_dF(WUNDER[F_UNDER], OUNDER[L2_UNDER]).squeeze()
	DG3UNDER_DB[F_UNDER] = dg3under_relu_dg3under[:,:,np.newaxis]
	DG3UNDER_DW[F_UNDER] = mult_partials(dg3under_relu_dg3under, dg3under_dw3under, OUNDER[F_UNDER].squeeze())
	
	dg3under_dg2under_relu = linear_F_dx(WUNDER[F_UNDER], OUNDER[L2_UNDER])
	dg2under_relu_dg2under = relu_dlayer_in(OUNDER[L2_UNDER])
	dg3under_dg2under = mult_partials(dg3under_dg2under_relu[:,:,np.newaxis], dg2under_relu_dg2under, OUNDER[L2_UNDER]).squeeze()
	dg2under_dw2under = linear_F_dF(WUNDER[L2_UNDER], OUNDER[L1_UNDER]).squeeze()
	dg3under_relu_dg2under = mult_partials(dg3under_relu_dg3under, dg3under_dg2under, OUNDER[F_UNDER].squeeze())
	DG3UNDER_DB[L2_UNDER] = dg3under_relu_dg2under[:,:,np.newaxis]
	DG3UNDER_DW[L2_UNDER] = mult_partials(dg3under_relu_dg2under, dg2under_dw2under, OUNDER[L2_UNDER].squeeze())

	dg2under_dg1under_relu = linear_F_dx(WUNDER[L2_UNDER], OUNDER[L1_UNDER])
	dg1under_relu_dg1under = relu_dlayer_in(OUNDER[L1_UNDER])
	dg2under_dg1under = mult_partials(dg2under_dg1under_relu[:,:,np.newaxis], dg1under_relu_dg1under, OUNDER[L1_UNDER]).squeeze()
	dg1under_dw1under = linear_F_dF(WUNDER[L1_UNDER], x).squeeze()
	dg2under_dw1under = mult_partials(dg2under_dg1under, dg1under_dw1under,  OUNDER[L1_UNDER].squeeze())
	DG3UNDER_DB[L1_UNDER] = mult_partials(dg3under_relu_dg2under, dg2under_dg1under, OUNDER[L2_UNDER].squeeze())[:,:,np.newaxis]
	DG3UNDER_DW[L1_UNDER] = mult_partials(dg3under_relu_dg2under, dg2under_dw1under, OUNDER[L2_UNDER].squeeze())
	
	return DG3UNDER_DW, DG3UNDER_DB
	
	
# gradients for layers underneath the read/write heads
# (used in mem_partials() and do_dw__inputs()
def dunder_gpu(L_WUNDER, L_BUNDER, L_OUNDER, X):
	
	DG3UNDER_DW = set_list_buffer([None] * len(L_OUNDER)) 
	DG3UNDER_DB = set_list_buffer([None] * len(L_OUNDER))
	DG3UNDER_RELU_DG3UNDER = init_buffer()
	DG3UNDER_DW3UNDER = init_buffer()
	DG3UNDER_DG2UNDER_RELU = init_buffer()
	DG2UNDER_RELU_DG2UNDER = init_buffer()
	DG3UNDER_DG2UNDER = init_buffer()
	DG2UNDER_DW2UNDER = init_buffer()
	DG3UNDER_RELU_DG2UNDER = init_buffer()
	DG2UNDER_DG1UNDER_RELU = init_buffer()
	DG1UNDER_RELU_DG1UNDER = init_buffer()
	DG2UNDER_DG1UNDER = init_buffer()
	DG1UNDER_DW1UNDER = init_buffer()
	DG2UNDER_DW1UNDER = init_buffer()
	
	nm.relu_dlayer_in(L_OUNDER[F_UNDER], DG3UNDER_RELU_DG3UNDER)
	nm.linear_F_dF(L_WUNDER[F_UNDER], L_OUNDER[L2_UNDER], DG3UNDER_DW3UNDER)
	DG3UNDER_DB[F_UNDER] = DG3UNDER_RELU_DG3UNDER
	nm.mult_partials(DG3UNDER_RELU_DG3UNDER, DG3UNDER_DW3UNDER, L_OUNDER[F_UNDER], DG3UNDER_DW[F_UNDER])
	
	nm.linear_F_dx(L_WUNDER[F_UNDER], L_OUNDER[L2_UNDER], DG3UNDER_DG2UNDER_RELU)
	nm.relu_dlayer_in(L_OUNDER[L2_UNDER], DG2UNDER_RELU_DG2UNDER)
	nm.mult_partials(DG3UNDER_DG2UNDER_RELU, DG2UNDER_RELU_DG2UNDER, L_OUNDER[L2_UNDER], DG3UNDER_DG2UNDER) ##
	nm.linear_F_dF(L_WUNDER[L2_UNDER], L_OUNDER[L1_UNDER], DG2UNDER_DW2UNDER)
	nm.mult_partials(DG3UNDER_RELU_DG3UNDER, DG3UNDER_DG2UNDER, L_OUNDER[F_UNDER], DG3UNDER_RELU_DG2UNDER, squeeze=1)
	DG3UNDER_DB[L2_UNDER] = DG3UNDER_RELU_DG2UNDER
	nm.mult_partials(DG3UNDER_RELU_DG2UNDER, DG2UNDER_DW2UNDER, L_OUNDER[L2_UNDER], DG3UNDER_DW[L2_UNDER], squeeze=1)
	
	nm.linear_F_dx(L_WUNDER[L2_UNDER], L_OUNDER[L1_UNDER], DG2UNDER_DG1UNDER_RELU)
	nm.relu_dlayer_in(L_OUNDER[L1_UNDER], DG1UNDER_RELU_DG1UNDER)
	nm.mult_partials(DG2UNDER_DG1UNDER_RELU, DG1UNDER_RELU_DG1UNDER, L_OUNDER[L1_UNDER], DG2UNDER_DG1UNDER)
	nm.linear_F_dF(L_WUNDER[L1_UNDER], X, DG1UNDER_DW1UNDER)
	nm.mult_partials(DG2UNDER_DG1UNDER, DG1UNDER_DW1UNDER,  L_OUNDER[L1_UNDER], DG2UNDER_DW1UNDER, squeeze=1)
	nm.mult_partials(DG3UNDER_RELU_DG2UNDER, DG2UNDER_DG1UNDER, L_OUNDER[L2_UNDER], DG3UNDER_DB[L1_UNDER], squeeze=1)
	nm.mult_partials(DG3UNDER_RELU_DG2UNDER, DG2UNDER_DW1UNDER, L_OUNDER[L2_UNDER], DG3UNDER_DW[L1_UNDER], squeeze=1)
	

	g3under_shape = np.asarray(L_OUNDER[F_UNDER][1])
	g3under_shape = tuple(g3under_shape[g3under_shape != 1])
	for i in range(len(L_WUNDER)):
		DG3UNDER_DW[i][1] = tuple(np.concatenate((g3under_shape, L_WUNDER[i][1])))
		DG3UNDER_DB[i][1] = tuple(np.concatenate((g3under_shape, L_BUNDER[i][1])))
	
	return DG3UNDER_DW, DG3UNDER_DB
	
	
#### intermediate gradients used in several places in do_dw__inputs() and do_dw__mem_prev()
def do_do_content_focused__(O, do_do_in):
	do_in_do_content_sm = interpolate_softmax_do_content(O[IN], O[IN_GATE], O[CONTENT_SM])
	do_content_sm_do_content_focused = nm.softmax_dlayer_in_cpu(O[CONTENT_SM])
	do_do_content_focused = mult_partials_chain((do_do_in, do_in_do_content_sm, do_content_sm_do_content_focused), (O[IN], O[CONTENT_SM]))
	
	return do_do_content_focused

# one step farther down the path from do_do_content_focused__()
# used in do_dw__inputs() and do_dw__mem_prev()
def do_do_content__(O, do_do_in):
	do_do_content_focused = do_do_content_focused__(O, do_do_in)
	do_content_focused_do_content = focus_key_dkeys(O[CONTENT], O[BETA])
	do_do_content = mult_partials(do_do_content_focused, do_content_focused_do_content, O[CONTENT_FOCUSED])
	
	return do_do_content

########## ...
# 25.2% of reverse_pass_partials()
#@profile
def do_dw__inputs(W, WUNDER, BUNDER, o_prev, OUNDER, DO_DWUNDER, DO_DBUNDER, O, DO_DW, DO_DB, mem_prev, x, do_do_in):
	DO_DW_NEW = copy.deepcopy(DO_DW); DO_DB_NEW = copy.deepcopy(DO_DB) # 3.5%
	DO_DWUNDER_NEW = copy.deepcopy(DO_DWUNDER); DO_DBUNDER_NEW = copy.deepcopy(DO_DBUNDER)
	
	## sharpen weights
	do_dgammarelu = nm.sharpen_dgamma_cpu(O[SHIFTED], O[GAMMA])
	dgammarelu_dgamma = relu_dlayer_in(O[GAMMA], thresh=1)
	do_dgamma = mult_partials(do_dgammarelu, dgammarelu_dgamma, O[GAMMA])
	DO_DB_NEW[GAMMA] += do_dgamma
	dgamma_dwgamma = linear_F_dF(W[GAMMA], OUNDER[F_UNDER])
	dgamma_dg3under = linear_F_dx(W[GAMMA], OUNDER[F_UNDER])
	DO_DW_NEW[GAMMA] += mult_partials(do_dgamma, dgamma_dwgamma, O[GAMMA])
	do_dg3under = np.squeeze(mult_partials(do_dgamma, dgamma_dg3under, O[GAMMA]))
	
	## shift weights
	do_dgshiftedsm = nm.sharpen_dw_cpu(O[SHIFTED], O[GAMMA])
	dgshiftedsm_dgshiftsm = shift_w_dshift_out(O[IN])
	do_dgshiftsm = mult_partials(do_dgshiftedsm, dgshiftedsm_dgshiftsm, O[SHARPENED])
	dgshiftsm_gshift = nm.softmax_dlayer_in_cpu(O[SHIFT])
	do_dgshift = mult_partials(do_dgshiftsm, dgshiftsm_gshift, O[SHIFT])
	DO_DB_NEW[SHIFT] += do_dgshift
	dgshift_dwshift = linear_2d_F_dF(W[SHIFT], OUNDER[F_UNDER])
	dgshift_dg3under = linear_2d_F_dx(W[SHIFT])
	DO_DW_NEW[SHIFT] += mult_partials(do_dgshift, dgshift_dwshift, O[SHIFT])
	do_dg3under += mult_partials(do_dgshift, dgshift_dg3under, O[SHIFT])
	
	## interp. gradients (wrt gin_gate)
	do_in_dgin_gate_sig = interpolate_softmax_dinterp_gate_out(O[IN], O[IN_GATE], O[CONTENT_SM], o_prev) # 4.2%
	do_dgin_gate_sig = mult_partials(do_do_in, do_in_dgin_gate_sig, O[IN])
	dgin_gate_sig_dgin_gate = sigmoid_dlayer_in(O[IN_GATE])
	do_dgin_gate = mult_partials(do_dgin_gate_sig, dgin_gate_sig_dgin_gate, O[IN_GATE])
	DO_DB_NEW[IN_GATE] += do_dgin_gate
	dgin_gate_dwin = linear_F_dF(W[IN_GATE], OUNDER[F_UNDER])
	dgin_gate_dg3under = linear_F_dx(W[IN_GATE], OUNDER[F_UNDER])
	DO_DW_NEW[IN_GATE] += mult_partials(do_dgin_gate, dgin_gate_dwin, O[IN_GATE])
	do_dg3under += np.squeeze(mult_partials(do_dgin_gate, dgin_gate_dg3under, O[IN_GATE]))
	
	## interp. gradients (wrt o_content; key)
	do_do_content = do_do_content__(O, do_do_in) # 14%
	do_content_dgkey = nm.cosine_sim_expand_dkeys_cpu(O[KEY], mem_prev) # 12.3%
	do_dgkey = mult_partials(do_do_content, do_content_dgkey, O[CONTENT])
	DO_DB_NEW[KEY] += do_dgkey
	dgkey_dwkey = linear_2d_F_dF(W[KEY], OUNDER[F_UNDER])
	dgkey_dg3under = linear_2d_F_dx(W[KEY])
	DO_DW_NEW[KEY] += mult_partials(do_dgkey, dgkey_dwkey, O[KEY]) # 28.6%
	do_dg3under += mult_partials(do_dgkey, dgkey_dg3under, O[KEY])
	
	## interp. gradients (wrt beta)
	do_do_content_focused = do_do_content_focused__(O, do_do_in) # 12.2%
	do_content_focused_dgbeta = focus_key_dbeta_out(O[CONTENT], O[BETA])
	do_dgbeta = mult_partials(do_do_content_focused, do_content_focused_dgbeta, O[CONTENT_FOCUSED])
	DO_DB_NEW[BETA] += do_dgbeta
	dgbeta_dwbeta = linear_F_dF(W[BETA], OUNDER[F_UNDER])
	dgbeta_dg3under = linear_F_dx(W[BETA], OUNDER[F_UNDER])
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
	do_content_dmem_prev = nm.cosine_sim_expand_dmem_cpu(O[KEY], mem_prev) # 15.5%
	do_dmem_prev = mult_partials(do_do_content, do_content_dmem_prev, O[CONTENT])
	
	DO_DW_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWW, mem_prev, DO_DW) # 49.6%
	DO_DB_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DBW, mem_prev, DO_DB)
	DO_DWUNDER_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DWUNDER, mem_prev, DO_DWUNDER)
	DO_DBUNDER_NEW = mult_partials__layers(do_dmem_prev, DMEM_PREV_DBUNDER, mem_prev, DO_DBUNDER)
	
	return DO_DW_NEW, DO_DB_NEW, DO_DWUNDER_NEW, DO_DBUNDER_NEW

# 18.6% of reverse_pass_partials()
def mem_partials(DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER,DMEM_PREV_DBUNDER, DOW_DWW, DOW_DBW,DOW_DWUNDER,DOW_DBUNDER,OW_PREV, OUNDER_PREV, WW,BW, WUNDER, BUNDER, x_prev, mem_prev_prev):
	
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
	derase_out_dwadd = linear_2d_F_dF(WW[ERASE], OUNDER_PREV[F_UNDER])
	DMEM_PREV_DWW_NEW[ERASE] += mult_partials(mem_prev_de_derase_out, derase_out_dwadd, OW_PREV[ERASE]) # de_dwadd
	
	# under: (wrt inputs)
	derase_out_dg3under = linear_2d_F_dx(WW[ERASE])
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
	dadd_out_dwadd = linear_2d_F_dF(WW[ADD], OUNDER_PREV[F_UNDER])
	DMEM_PREV_DWW_NEW[ADD] += mult_partials(da_dadd_out, dadd_out_dwadd, OW_PREV[ADD]) # da_dwadd
	
	# under: (wrt inputs)
	dadd_out_dg3under = linear_2d_F_dx(WW[ADD])
	da_dg3under = mult_partials(da_dadd_out, dadd_out_dg3under, OW_PREV[ADD]) # da_dwunder
	
	DMEM_PREV_DWUNDER_NEW = mult_partials__layers(da_dg3under, DG3UNDER_DW, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DWUNDER_NEW)
	DMEM_PREV_DBUNDER_NEW = mult_partials__layers(da_dg3under, DG3UNDER_DB, np.squeeze(OUNDER_PREV[F_UNDER]), DMEM_PREV_DBUNDER_NEW)
	
	return DMEM_PREV_DWW_NEW, DMEM_PREV_DBW_NEW, DMEM_PREV_DWUNDER_NEW, DMEM_PREV_DBUNDER_NEW
	
	
	
def mem_partials_gpu(DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER,DMEM_PREV_DBUNDER, DOW_DWW, DOW_DBW,DOW_DWUNDER,DOW_DBUNDER,OW_PREV, OUNDER_PREV, WW,BW, WUNDER, BUNDER, x_prev, mem_prev_prev):
	
	L_WUNDER = set_list_buffer(WUNDER)
	L_BUNDER = set_list_buffer(BUNDER)
	L_OUNDER_PREV = set_list_buffer(OUNDER_PREV)
	L_OW_PREV = set_list_buffer(OW_PREV)
	L_DMEM_PREV_DWW = set_list_buffer(DMEM_PREV_DWW)
	L_DMEM_PREV_DBW = set_list_buffer(DMEM_PREV_DBW)
	L_DMEM_PREV_DWUNDER = set_list_buffer(DMEM_PREV_DWUNDER)
	L_DMEM_PREV_DBUNDER = set_list_buffer(DMEM_PREV_DBUNDER)
	
	L_WW = set_list_buffer(WW)
	L_BW = set_list_buffer(BW)
	L_WUNDER = set_list_buffer(WUNDER)
	L_BUNDER = set_list_buffer(BUNDER)
	
	LO_DMEM_PREV_DWW = set_list_buffer(DMEM_PREV_DWW)
	LO_DMEM_PREV_DBW = set_list_buffer(DMEM_PREV_DBW)
	LO_DMEM_PREV_DWUNDER = set_list_buffer(DMEM_PREV_DWUNDER)
	LO_DMEM_PREV_DBUNDER = set_list_buffer(DMEM_PREV_DBUNDER)
	
	L_DOW_DWW = set_list_buffer(DOW_DWW)
	L_DOW_DBW = set_list_buffer(DOW_DBW)
	L_DOW_DWUNDER = set_list_buffer(DOW_DWUNDER)
	L_DOW_DBUNDER = set_list_buffer(DOW_DBUNDER)
	
	X_PREV = init_buffer(x_prev)
	MEM_PREV_PREV = init_buffer(mem_prev_prev)
	E = init_buffer()
	MEM_PREV_TIMES_DE_DOW = init_buffer()
	MEM_PREV_DE_DERASE_OUT_SIG = init_buffer()
	DERASE_OUT_SIG_DERASE_OUT = init_buffer()
	MEM_PREV_DE_DERASE_OUT = init_buffer()
	DERASE_OUT_DWADD = init_buffer()
	DERASE_OUT_DG3UNDER = init_buffer()
	MEM_PREV_DE_DG3UNDER = init_buffer()
	DA_DADD_OUT = init_buffer()
	DA_DOW = init_buffer()
	DADD_OUT_DWADD = init_buffer()
	DADD_OUT_DG3UNDER = init_buffer()
	DA_DG3UNDER = init_buffer()
	
	DG3UNDER_DW, DG3UNDER_DB = dunder_gpu(L_WUNDER, L_BUNDER, L_OUNDER_PREV, X_PREV)
	
	####################
	###########DG3UNDER_DW, DG3UNDER_DB = dunder(WUNDER, BUNDER, OUNDER_PREV, x_prev)
	# mem = mem_prev*(1 - e) + a
	# dmem = dmem_prev*(1 - e) - mem_prev*de + da
	
	# write gradients (erase)
	nm.add_mem(L_OW_PREV[F], L_OW_PREV[ERASE], E)
	
	nm.add_mem_dgw(L_OW_PREV[F], L_OW_PREV[ERASE], MEM_PREV_TIMES_DE_DOW) # de_dow
	nm.point_wise_mult_bcast2(MEM_PREV_TIMES_DE_DOW, MEM_PREV_PREV, scalar=-1) # -mem_prev * de
	nm.point_wise_add_scalar(E, scalar1= -1, scalar2=1) # 1 - e
	
	# dmem_prev * (1 - e)
	nm.pointwise_mult_partials__layers(L_DMEM_PREV_DWW, E)
	nm.pointwise_mult_partials__layers(L_DMEM_PREV_DBW, E)
	nm.pointwise_mult_partials__layers(L_DMEM_PREV_DWUNDER, E)
	nm.pointwise_mult_partials__layers(L_DMEM_PREV_DBUNDER, E)

	# dmem_prev * (1 - e) - mem_prev * de
	nm.mult_partials__layers(MEM_PREV_TIMES_DE_DOW, L_DOW_DWW, L_OW_PREV[F], L_DMEM_PREV_DWW)
	nm.mult_partials__layers(MEM_PREV_TIMES_DE_DOW, L_DOW_DBW, L_OW_PREV[F], L_DMEM_PREV_DBW)
	nm.mult_partials__layers(MEM_PREV_TIMES_DE_DOW, L_DOW_DWUNDER, L_OW_PREV[F], L_DMEM_PREV_DWUNDER)
	nm.mult_partials__layers(MEM_PREV_TIMES_DE_DOW, L_DOW_DBUNDER, L_OW_PREV[F], L_DMEM_PREV_DBUNDER)
	
	# W[ERASE] gradients (de wrt W[ERASE])
	nm.add_mem_dadd_out(L_OW_PREV[F], L_OW_PREV[ERASE], MEM_PREV_DE_DERASE_OUT_SIG)
	nm.point_wise_mult_bcast2(MEM_PREV_DE_DERASE_OUT_SIG, MEM_PREV_PREV, scalar=-1) # -mem_prev * de
	nm.sigmoid_dlayer_in(L_OW_PREV[ERASE], DERASE_OUT_SIG_DERASE_OUT)
	nm.mult_partials(MEM_PREV_DE_DERASE_OUT_SIG, DERASE_OUT_SIG_DERASE_OUT, L_OW_PREV[ERASE], MEM_PREV_DE_DERASE_OUT)
	nm.point_wise_add(L_DMEM_PREV_DBW[ERASE], MEM_PREV_DE_DERASE_OUT)
	nm.linear_2d_F_dF(L_WW[ERASE], L_OUNDER_PREV[F_UNDER], DERASE_OUT_DWADD)
	DERASE_OUT_DWADD[1] = (np.prod(np.asarray(DERASE_OUT_DWADD[1])[:2]),np.prod(np.asarray(DERASE_OUT_DWADD[1])[2:]))
	L_OW_PREV[ERASE][1] = (np.prod(np.asarray(L_OW_PREV[ERASE][1])),)
	nm.mult_partials(MEM_PREV_DE_DERASE_OUT, DERASE_OUT_DWADD, L_OW_PREV[ERASE], L_DMEM_PREV_DWW[ERASE], increment=1) # de_dwadd
	
	# under: (wrt inputs)
	nm.linear_2d_F_dx(L_WW[ERASE], L_OUNDER_PREV[F_UNDER], DERASE_OUT_DG3UNDER)
	DERASE_OUT_DG3UNDER[1] = (np.prod(np.asarray(DERASE_OUT_DG3UNDER[1])[:2]),np.prod(np.asarray(DERASE_OUT_DG3UNDER[1])[2:]))
	nm.mult_partials(MEM_PREV_DE_DERASE_OUT, DERASE_OUT_DG3UNDER, L_OW_PREV[ERASE], MEM_PREV_DE_DG3UNDER)
	for layer in range(len(L_DMEM_PREV_DWUNDER)):
		sz = np.prod(np.asarray(L_DMEM_PREV_DWUNDER[layer][1]))
		L_DMEM_PREV_DWUNDER[layer][1] = tuple(np.concatenate((sz / L_WUNDER[layer][1], np.asarray(L_WUNDER[layer][1]))))
	nm.mult_partials__layers(MEM_PREV_DE_DG3UNDER, DG3UNDER_DW, L_OUNDER_PREV[F_UNDER], L_DMEM_PREV_DWUNDER, increment=1, squeeze=1)
	nm.mult_partials__layers(MEM_PREV_DE_DG3UNDER, DG3UNDER_DB, L_OUNDER_PREV[F_UNDER], L_DMEM_PREV_DBUNDER, increment=1, squeeze=1)
	
	################
	# write gradients (add)
	nm.add_mem_dgw(L_OW_PREV[F], L_OW_PREV[ADD], DA_DOW) # da
	nm.mult_partials__layers(DA_DOW, L_DOW_DWW, L_OW_PREV[F], L_DMEM_PREV_DWW, increment=1) # da_dlayer
	nm.mult_partials__layers(DA_DOW, L_DOW_DBW, L_OW_PREV[F], L_DMEM_PREV_DBW, increment=1) # da_dlayer
	nm.mult_partials__layers(DA_DOW, L_DOW_DWUNDER, L_OW_PREV[F], L_DMEM_PREV_DWUNDER, increment=1)
	nm.mult_partials__layers(DA_DOW, L_DOW_DBUNDER, L_OW_PREV[F], L_DMEM_PREV_DBUNDER, increment=1)
	
	###
	# W[ADD] gradients
	nm.add_mem_dadd_out(L_OW_PREV[F], L_OW_PREV[ADD], DA_DADD_OUT)
	
	nm.point_wise_add(L_DMEM_PREV_DBW[ADD], DA_DADD_OUT)
	nm.linear_2d_F_dF(L_WW[ADD], L_OUNDER_PREV[F_UNDER], DADD_OUT_DWADD)
	nm.mult_partials(DA_DADD_OUT, DADD_OUT_DWADD, L_OW_PREV[ADD], L_DMEM_PREV_DWW[ADD], increment=1) # da_dwadd
	
	# under: (wrt inputs)
	nm.linear_2d_F_dx(L_WW[ADD], L_OUNDER_PREV[F_UNDER], DADD_OUT_DG3UNDER)
	nm.mult_partials(DA_DADD_OUT, DADD_OUT_DG3UNDER, L_OW_PREV[ADD], DA_DG3UNDER) # da_dwunder
	
	nm.mult_partials__layers(DA_DG3UNDER, DG3UNDER_DW, L_OUNDER_PREV[F_UNDER], L_DMEM_PREV_DWUNDER, increment=1, squeeze=1)
	nm.mult_partials__layers(DA_DG3UNDER, DG3UNDER_DB, L_OUNDER_PREV[F_UNDER], L_DMEM_PREV_DBUNDER, increment=1, squeeze=1)
	
	DMEM_PREV_DWW_NEW = return_list_buffer(L_DMEM_PREV_DWW, LO_DMEM_PREV_DWW)
	DMEM_PREV_DBW_NEW = return_list_buffer(L_DMEM_PREV_DBW, LO_DMEM_PREV_DBW)
	DMEM_PREV_DWUNDER_NEW = return_list_buffer(L_DMEM_PREV_DWUNDER, LO_DMEM_PREV_DWUNDER)
	DMEM_PREV_DBUNDER_NEW = return_list_buffer(L_DMEM_PREV_DBUNDER, LO_DMEM_PREV_DBUNDER)
	
	return DMEM_PREV_DWW_NEW, DMEM_PREV_DBW_NEW, DMEM_PREV_DWUNDER_NEW, DMEM_PREV_DBUNDER_NEW

### compute state partials (stores history, in a sense) 
# 74.7% of main()
#@profile
def reverse_pass_partials(WUNDER,BUNDER, WR,WW,BR,BW, OUNDER, OUNDER_PREV, OR, OR_PREV, OW_PREV, OW_PREV_PREV, mem_prev, mem_prev_prev, x, x_prev, frame, DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW):
	dor_dgsharpen = nm.sharpen_dw_cpu(OR[SHIFTED], OR[GAMMA])
	dow_prev_dgsharpen = nm.sharpen_dw_cpu(OW_PREV[SHIFTED], OW_PREV[GAMMA])
	
	dgsharpen_dor_in = shift_w_dw_interp(OR[SHIFT])
	dgsharpen_dow_prev_in = shift_w_dw_interp(OW_PREV[SHIFT])
	
	dor_dor_in = mult_partials(dor_dgsharpen, dgsharpen_dor_in, OR[SHARPENED])
	dow_prev_dow_prev_in = mult_partials(dow_prev_dgsharpen, dgsharpen_dow_prev_in, OW_PREV[SHARPENED])
	
	# partials for write head output (OW)
	if frame > 1:
		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__o_prev(WW, OW_PREV_PREV[F], DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, dow_prev_dow_prev_in) # 13.4%
		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__mem_prev(WW, DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, mem_prev_prev, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER,  dow_prev_dow_prev_in) # 10.3%
		DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER = do_dw__inputs(WW, WUNDER, BUNDER, OW_PREV_PREV[F], OUNDER_PREV, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, DOW_DWW, DOW_DBW, mem_prev_prev, x_prev, dow_prev_dow_prev_in) # 12.6%
		
		####
		#DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER = mem_partials(DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, OUNDER_PREV, WW, BW, WUNDER, BUNDER, x_prev, mem_prev_prev) # 18.6%
		
		DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER = mem_partials_gpu(DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, OW_PREV, OUNDER_PREV, WW, BW, WUNDER, BUNDER, x_prev, mem_prev_prev) # 18.6%
	
	# partials from read head output (OR)
	DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER = do_dw__o_prev(WR, OR_PREV[F], DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, OR, dor_dor_in) # 7.8%
	DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER = do_dw__inputs(WR, WUNDER, BUNDER, OR_PREV[F], OUNDER, DOR_DWUNDER, DOR_DBUNDER, OR, DOR_DWR, DOR_DBR, mem_prev, x, dor_dor_in) # 12.3%
	
	DOR_DWW, DOR_DBW = do_dw__o_prev(WR, OR_PREV[F], DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, dor_dor_in)[:2] #?... 13.4%
	DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER = do_dw__mem_prev(WR, DOR_DWW, DOR_DBW, DOR_DWUNDER, DOR_DBUNDER, OR, mem_prev, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, dor_dor_in) # 10.4%
	
	return DOW_DWW, DOW_DBW, DOW_DWUNDER, DOW_DBUNDER, DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER, DOR_DWR, DOR_DBR, DOR_DWUNDER, DOR_DBUNDER, DOR_DWW, DOR_DBW



### compute full gradients from state partials
def full_gradients_gpu(READ_MEM, T, MEM_PREV, L_DOR_DWR, L_DOR_DBR, L_DOR_DWW, L_DOR_DBW, L_DOR_DWUNDER, L_DOR_DBUNDER, L_OR, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, L_OABOVE, L_WABOVE):
	
	DIFF_OUT = init_buffer()
	DERR_DG2ABOVE_RELU = init_buffer()
	DG2ABOVE_RELU_DG2ABOVE = init_buffer()
	DERR_DG2ABOVE = init_buffer()
	DG2ABOVE_DG1ABOVE_RELU = init_buffer()
	DG1ABOVE_RELU_DG1ABOVE = init_buffer()
	DG1ABOVE_DREAD_MEM = init_buffer()
	DG2ABOVE_DW2ABOVE = init_buffer()
	DG1ABOVE_DW1ABOVE = init_buffer()
	DERR_DREAD_MEM = init_buffer()
	DREAD_MEM_DOR = init_buffer()
	DERR_DOR = init_buffer()
	DREAD_MEM_DMEM_PREV = init_buffer()
	DERR_DMEM_PREV = init_buffer()
	
	L_DERR_DG1ABOVE = set_list_buffer([None]*3)
	L_DWR = set_list_buffer([None]*len(L_DOR_DWR))
	L_DBR = set_list_buffer([None]*len(L_DOR_DBR))
	L_DWW = set_list_buffer([None]*len(L_DOR_DWW))
	L_DBW = set_list_buffer([None]*len(L_DOR_DBW))
	L_DWABOVE = set_list_buffer([None]*len(L_OABOVE))
	L_DBABOVE = set_list_buffer([None]*len(L_OABOVE))
	L_DWUNDER = set_list_buffer([None]*len(L_DOR_DWUNDER))
	L_DBUNDER = set_list_buffer([None]*len(L_DOR_DBUNDER))
	
	READ_MEM_reshape = copy.deepcopy(READ_MEM)
	READ_MEM_reshape[1] = (C*mem_length,1)
	
	# above the read/write heads
	nm.point_wise_add(L_OABOVE[F_ABOVE], T, -1, DIFF_OUT)
	nm.sq_points_dinput(DIFF_OUT, DERR_DG2ABOVE_RELU)
	
	nm.relu_dlayer_in(L_OABOVE[F_ABOVE], DG2ABOVE_RELU_DG2ABOVE)
	nm.mult_partials(DERR_DG2ABOVE_RELU, DG2ABOVE_RELU_DG2ABOVE, L_OABOVE[F_ABOVE], DERR_DG2ABOVE)
	
	nm.linear_F_dx(L_WABOVE[F_ABOVE], L_OABOVE[L1_ABOVE], DG2ABOVE_DG1ABOVE_RELU)
	nm.relu_dlayer_in(L_OABOVE[L1_ABOVE], DG1ABOVE_RELU_DG1ABOVE)
	nm.linear_F_dx(L_WABOVE[L1_ABOVE], READ_MEM_reshape, DG1ABOVE_DREAD_MEM)
	nm.mult_partials_chain((DERR_DG2ABOVE, DG2ABOVE_DG1ABOVE_RELU, DG1ABOVE_RELU_DG1ABOVE), (L_OABOVE[F_ABOVE], L_OABOVE[L1_ABOVE]), L_DERR_DG1ABOVE)
	DERR_DG1ABOVE = L_DERR_DG1ABOVE[-1]
	
	# above weight gradients
	nm.linear_F_dF(L_WABOVE[F_ABOVE], L_OABOVE[L1_ABOVE], DG2ABOVE_DW2ABOVE)
	nm.linear_F_dF(L_WABOVE[L1_ABOVE], READ_MEM_reshape, DG1ABOVE_DW1ABOVE)
	
	nm.mult_partials(DERR_DG2ABOVE, DG2ABOVE_DW2ABOVE, L_OABOVE[F_ABOVE], L_DWABOVE[F_ABOVE])
	nm.mult_partials(DERR_DG1ABOVE, DG1ABOVE_DW1ABOVE, L_OABOVE[L1_ABOVE], L_DWABOVE[L1_ABOVE])
	
	L_DBABOVE[F_ABOVE] = DERR_DG2ABOVE
	L_DBABOVE[L1_ABOVE] = DERR_DG1ABOVE
	
	# read weights
	nm.mult_partials(DERR_DG1ABOVE, DG1ABOVE_DREAD_MEM, L_OABOVE[L1_ABOVE], DERR_DREAD_MEM)
	nm.linear_F_dF(L_OR[F], MEM_PREV, DREAD_MEM_DOR)
	nm.mult_partials(DERR_DREAD_MEM, DREAD_MEM_DOR, READ_MEM, DERR_DOR)
	
	nm.mult_partials__layers(DERR_DOR, L_DOR_DWR, L_OR[F], L_DWR)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DBR, L_OR[F], L_DBR)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DWW, L_OR[F], L_DWW)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DBW, L_OR[F], L_DBW)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DWUNDER, L_OR[F], L_DWUNDER)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DBUNDER, L_OR[F], L_DBUNDER)
	
	# write weights
	nm.linear_F_dx(L_OR[F], MEM_PREV, DREAD_MEM_DMEM_PREV)
	nm.mult_partials(DERR_DREAD_MEM, DREAD_MEM_DMEM_PREV, READ_MEM, DERR_DMEM_PREV)
	
	nm.mult_partials__layers(DERR_DMEM_PREV, L_DMEM_PREV_DWW, MEM_PREV, L_DWW, increment=1)
	nm.mult_partials__layers(DERR_DMEM_PREV, L_DMEM_PREV_DBW, MEM_PREV, L_DBW, increment=1)
	nm.mult_partials__layers(DERR_DMEM_PREV, L_DMEM_PREV_DWUNDER, MEM_PREV, L_DWUNDER, increment=1)
	nm.mult_partials__layers(DERR_DMEM_PREV, L_DMEM_PREV_DBUNDER, MEM_PREV, L_DBUNDER, increment=1)
	
	return L_DWR, L_DBR, L_DWW, L_DBW, L_DWUNDER, L_DBUNDER, L_DWABOVE, L_DBABOVE

