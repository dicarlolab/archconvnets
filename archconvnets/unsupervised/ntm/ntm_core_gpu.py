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
from archconvnets.unsupervised.ntm_module.ntm_module import init_buffer, set_list_buffer, return_list_buffer, return_buffer

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
def do_do_content_focused__(L_O, DO_DO_IN, O_PREV):
	DO_IN_DO_CONTENT_SM = init_buffer()
	DO_CONTENT_SM_DO_CONTENT_FOCUSED = init_buffer()
	DO_DO_CONTENT_FOCUSED = init_buffer()
	DO_DO_CONTENT_SM = init_buffer()
	
	L_O[IN][1] = O_PREV[1]
	nm.interpolate_softmax_do_content(L_O[IN], L_O[IN_GATE], O_PREV, DO_IN_DO_CONTENT_SM)
	nm.softmax_dlayer_in(L_O[CONTENT_SM], DO_CONTENT_SM_DO_CONTENT_FOCUSED)
	DO_IN_DO_CONTENT_SM[1] = DO_DO_IN[1]
	L_O[IN][1] = (L_O[IN][1][0]*L_O[IN][1][1],)
	nm.mult_partials(DO_DO_IN, DO_IN_DO_CONTENT_SM, L_O[IN], DO_DO_CONTENT_SM)
	DO_DO_CONTENT_SM[1] = DO_CONTENT_SM_DO_CONTENT_FOCUSED[1]
	nm.mult_partials(DO_DO_CONTENT_SM, DO_CONTENT_SM_DO_CONTENT_FOCUSED, L_O[CONTENT_SM], DO_DO_CONTENT_FOCUSED)
	
	return DO_DO_CONTENT_FOCUSED

# one step farther down the path from do_do_content_focused__()
# used in do_dw__inputs() and do_dw__mem_prev()
def do_do_content__(L_O, DO_DO_IN, O_PREV):
	DO_CONTENT_FOCUSED_DO_CONTENT = init_buffer()
	DO_DO_CONTENT = init_buffer()
	
	DO_DO_CONTENT_FOCUSED = do_do_content_focused__(L_O, DO_DO_IN, O_PREV)
	nm.focus_key_dkeys(L_O[BETA], L_O[CONTENT], DO_CONTENT_FOCUSED_DO_CONTENT)
	DO_DO_CONTENT_FOCUSED[1] = DO_CONTENT_FOCUSED_DO_CONTENT[1]
	nm.mult_partials(DO_DO_CONTENT_FOCUSED, DO_CONTENT_FOCUSED_DO_CONTENT, L_O[CONTENT_FOCUSED], DO_DO_CONTENT)
	
	return DO_DO_CONTENT

########## ...
# 25.2% of reverse_pass_partials()
#@profile
def do_dw__inputs_gpu(L_W, L_WUNDER, L_BUNDER, O_PREV, L_OUNDER, L_DO_DWUNDER, L_DO_DBUNDER, L_O, L_DO_DW, L_DO_DB, MEM_PREV, X, DO_DO_IN):
	DO_DGAMMARELU = init_buffer()
	DGAMMARELU_DGAMMA = init_buffer()
	DO_DGAMMA = init_buffer()
	DO_DG3UNDER = init_buffer()
	DGAMMA_DWGAMMA = init_buffer()
	DGAMMA_DG3UNDER = init_buffer()
	DO_DGSHIFTEDSM = init_buffer()
	DGSHIFTEDSM_DGSHIFTSM = init_buffer()
	DGSHIFTSM_GSHIFT = init_buffer()
	DO_DGSHIFT = init_buffer()
	DGSHIFT_DWSHIFT = init_buffer()
	DO_DGSHIFTSM = init_buffer()
	DGSHIFT_DG3UNDER = init_buffer()
	DO_IN_DGIN_GATE_SIG = init_buffer()
	DO_DGIN_GATE_SIG = init_buffer()
	DO_DGIN_GATE = init_buffer()
	DGIN_GATE_DWIN = init_buffer()
	DGIN_GATE_DG3UNDER = init_buffer()
	DO_CONTENT_DGKEY = init_buffer()
	DO_DGKEY = init_buffer()
	DGKEY_DWKEY = init_buffer()
	DO_CONTENT_FOCUSED_DGBETA = init_buffer()
	DO_DGBETA = init_buffer()
	DGBETA_DWBETA = init_buffer()
	DGBETA_DG3UNDER = init_buffer()
	DGIN_GATE_SIG_DGIN_GATE = init_buffer()
	DGKEY_DG3UNDER = init_buffer()
	
	## sharpen weights
	nm.sharpen_dgamma(L_O[SHIFTED], L_O[GAMMA], DO_DGAMMARELU)
	nm.relu_dlayer_in(L_O[GAMMA], DGAMMARELU_DGAMMA, thresh=1)
	nm.mult_partials(DO_DGAMMARELU, DGAMMARELU_DGAMMA, L_O[GAMMA], DO_DGAMMA)
	nm.point_wise_add(L_DO_DB[GAMMA], DO_DGAMMA)
	nm.linear_F_dF(L_W[GAMMA], L_OUNDER[F_UNDER], DGAMMA_DWGAMMA)
	nm.linear_F_dx(L_W[GAMMA], L_OUNDER[F_UNDER], DGAMMA_DG3UNDER)
	nm.mult_partials(DO_DGAMMA, DGAMMA_DWGAMMA, L_O[GAMMA], L_DO_DW[GAMMA], squeeze=1, increment=1)
	nm.mult_partials(DO_DGAMMA, DGAMMA_DG3UNDER, L_O[GAMMA], DO_DG3UNDER, squeeze=1)
	
	## shift weights
	nm.sharpen_dw(L_O[SHIFTED], L_O[GAMMA], DO_DGSHIFTEDSM)
	L_O[IN][1] = L_O[SHIFTED][1]
	nm.shift_w_dshift_out(L_O[IN], DGSHIFTEDSM_DGSHIFTSM)
	nm.mult_partials(DO_DGSHIFTEDSM, DGSHIFTEDSM_DGSHIFTSM, L_O[SHARPENED], DO_DGSHIFTSM)
	nm.softmax_dlayer_in(L_O[SHIFT], DGSHIFTSM_GSHIFT)
	DO_DGSHIFTSM[1] = (DO_DGSHIFTSM[1][0], L_O[SHIFT][1][0], L_O[SHIFT][1][1])
	nm.mult_partials(DO_DGSHIFTSM, DGSHIFTSM_GSHIFT, L_O[SHIFT], DO_DGSHIFT)
	nm.point_wise_add(L_DO_DB[SHIFT], DO_DGSHIFT)
	nm.linear_2d_F_dF(L_W[SHIFT], L_OUNDER[F_UNDER], DGSHIFT_DWSHIFT)
	nm.linear_2d_F_dx(L_W[SHIFT], L_OUNDER[F_UNDER], DGSHIFT_DG3UNDER)
	DO_DGSHIFT[1] = (DO_DGSHIFT[1][0], L_O[SHIFT][1][0], L_O[SHIFT][1][1])
	nm.mult_partials(DO_DGSHIFT, DGSHIFT_DWSHIFT, L_O[SHIFT], L_DO_DW[SHIFT], increment=1)
	nm.mult_partials(DO_DGSHIFT, DGSHIFT_DWSHIFT, L_O[SHIFT], L_DO_DW[SHIFT], increment=1)
	nm.mult_partials(DO_DGSHIFT, DGSHIFT_DG3UNDER, L_O[SHIFT], DO_DG3UNDER, increment=1)
	
	## interp. gradients (wrt gin_gate)
	nm.interpolate_softmax_dinterp_gate_out(L_O[IN], L_O[CONTENT_SM], O_PREV, DO_IN_DGIN_GATE_SIG) # 4.2%
	L_O[IN][1] = (L_O[IN][1][0]*L_O[IN][1][1],)
	nm.mult_partials(DO_DO_IN, DO_IN_DGIN_GATE_SIG, L_O[IN], DO_DGIN_GATE_SIG)
	nm.sigmoid_dlayer_in(L_O[IN_GATE], DGIN_GATE_SIG_DGIN_GATE)
	nm.mult_partials(DO_DGIN_GATE_SIG, DGIN_GATE_SIG_DGIN_GATE, L_O[IN_GATE], DO_DGIN_GATE, squeeze=1)
	nm.point_wise_add(L_DO_DB[IN_GATE], DO_DGIN_GATE)
	nm.linear_F_dF(L_W[IN_GATE], L_OUNDER[F_UNDER], DGIN_GATE_DWIN)
	nm.linear_F_dx(L_W[IN_GATE], L_OUNDER[F_UNDER], DGIN_GATE_DG3UNDER)
	nm.mult_partials(DO_DGIN_GATE, DGIN_GATE_DWIN, L_O[IN_GATE], L_DO_DW[IN_GATE], increment=1, squeeze=1)
	nm.mult_partials(DO_DGIN_GATE, DGIN_GATE_DG3UNDER, L_O[IN_GATE], DO_DG3UNDER, increment=1, squeeze=1)
	
	## interp. gradients (wrt o_content; key)
	DO_DO_CONTENT = do_do_content__(L_O, DO_DO_IN, O_PREV) # 14%
	nm.cosine_sim_expand_dkeys(L_O[KEY], MEM_PREV, DO_CONTENT_DGKEY) # 12.3%
	DO_DO_CONTENT[1] = (L_O[CONTENT][1][0], L_O[CONTENT][1][1], L_O[CONTENT][1][0], L_O[CONTENT][1][1]) 
	nm.mult_partials(DO_DO_CONTENT, DO_CONTENT_DGKEY, L_O[CONTENT], DO_DGKEY)
	nm.point_wise_add(L_DO_DB[KEY], DO_DGKEY)
	nm.linear_2d_F_dF(L_W[KEY], L_OUNDER[F_UNDER], DGKEY_DWKEY)
	nm.linear_2d_F_dx(L_W[KEY], L_OUNDER[F_UNDER], DGKEY_DG3UNDER)
	#print DO_DGKEY[1], DGKEY_DWKEY[1], L_O[KEY][1], L_DO_DW[KEY][1]
	DO_DGKEY[1] = (DO_DGKEY[1][0], L_O[KEY][1][0], L_O[KEY][1][1])
	#print DO_DGKEY[1], DGKEY_DWKEY[1], L_O[KEY][1], L_DO_DW[KEY][1]
	nm.mult_partials(DO_DGKEY, DGKEY_DWKEY, L_O[KEY], L_DO_DW[KEY], increment=1) # 28.6%
	nm.mult_partials(DO_DGKEY, DGKEY_DG3UNDER, L_O[KEY], DO_DG3UNDER, increment=1)
	
	## interp. gradients (wrt beta)
	DO_DO_CONTENT_FOCUSED = do_do_content_focused__(L_O, DO_DO_IN, O_PREV) # 12.2%
	nm.focus_key_dbeta_out(L_O[CONTENT], DO_CONTENT_FOCUSED_DGBETA)
	DO_DO_CONTENT_FOCUSED[1] = (L_O[CONTENT_FOCUSED][1][0], L_O[CONTENT_FOCUSED][1][1], L_O[CONTENT_FOCUSED][1][0], L_O[CONTENT_FOCUSED][1][1])
	nm.mult_partials(DO_DO_CONTENT_FOCUSED, DO_CONTENT_FOCUSED_DGBETA, L_O[CONTENT_FOCUSED], DO_DGBETA, squeeze=1)
	nm.point_wise_add(L_DO_DB[BETA], DO_DGBETA)
	nm.linear_F_dF(L_W[BETA], L_OUNDER[F_UNDER], DGBETA_DWBETA)
	nm.linear_F_dx(L_W[BETA], L_OUNDER[F_UNDER], DGBETA_DG3UNDER)
	#print DO_DGBETA[1], DGBETA_DWBETA[1], L_O[BETA][1], L_DO_DW[BETA][1]
	#DO_DGBETA[1] = (L_DO_DW[BETA][1][0], L_DO_DW[BETA][1][1], DO_DGBETA[1][1])
	#print DO_DGBETA[1], DGBETA_DWBETA[1], L_O[BETA][1], L_DO_DW[BETA][1]
	nm.mult_partials(DO_DGBETA, DGBETA_DWBETA, L_O[BETA], L_DO_DW[BETA], increment=1, squeeze=1)
	nm.mult_partials(DO_DGBETA, DGBETA_DG3UNDER, L_O[BETA], DO_DG3UNDER, increment=1, squeeze=1)
	
	## combine weights under gradients
	L_DG3UNDER_DW, L_DG3UNDER_DB = dunder_gpu(L_WUNDER, L_BUNDER, L_OUNDER, X)
	
	nm.mult_partials__layers(DO_DG3UNDER, L_DG3UNDER_DW, L_OUNDER[F_UNDER], L_DO_DWUNDER, increment=1, squeeze=1)
	nm.mult_partials__layers(DO_DG3UNDER, L_DG3UNDER_DB, L_OUNDER[F_UNDER], L_DO_DBUNDER, increment=1, squeeze=1)
	
	return L_DO_DW, L_DO_DB, L_DO_DWUNDER, L_DO_DBUNDER

########## ...
# 34.2% of reverse_pass_partials()
def do_dw__o_prev(L_W, O_PREV, L_DO_DW, L_DO_DB, L_DO_DWUNDER,L_DO_DBUNDER, L_O, DO_DO_IN):
	DO_IN_DO_PREV = init_buffer()
	DO_DO_PREV = init_buffer()
	
	L_O[IN][1] = O_PREV[1]
	nm.interpolate_softmax_do_prev(L_O[IN], L_O[IN_GATE], O_PREV, DO_IN_DO_PREV) # 6.2%
	L_O[IN][1] = (L_O[IN][1][0]*L_O[IN][1][1],)
	nm.mult_partials(DO_DO_IN, DO_IN_DO_PREV, L_O[IN], DO_DO_PREV)
	
	L_DO_DW_NEW = set_list_buffer(return_list_buffer(L_DO_DW))
	L_DO_DB_NEW = set_list_buffer(return_list_buffer(L_DO_DB))
	L_DO_DWUNDER_NEW = set_list_buffer(return_list_buffer(L_DO_DWUNDER))
	L_DO_DBUNDER_NEW = set_list_buffer(return_list_buffer(L_DO_DBUNDER))
	
	DO_DO_PREV[1] = (O_PREV[1][0], O_PREV[1][1], O_PREV[1][0], O_PREV[1][1])
	nm.mult_partials__layers(DO_DO_PREV, L_DO_DW, O_PREV, L_DO_DW_NEW) # 67.9%
	nm.mult_partials__layers(DO_DO_PREV, L_DO_DB, O_PREV, L_DO_DB_NEW) # 8.8%
	L_DO_DWUNDER[0][1] = (16, 6, 20, 2)
	L_DO_DWUNDER[1][1] = (16, 6, 22, 20)
	L_DO_DWUNDER[2][1] = (16, 6, 9, 22)
	nm.mult_partials__layers(DO_DO_PREV, L_DO_DWUNDER, O_PREV, L_DO_DWUNDER_NEW) # 13.3%
	L_DO_DBUNDER[0][1] = (16, 6, 20, 1)
	L_DO_DBUNDER[1][1] = (16, 6, 22, 1)
	L_DO_DBUNDER[2][1] = (16, 6, 9, 1)
	nm.mult_partials__layers(DO_DO_PREV, L_DO_DBUNDER, O_PREV, L_DO_DBUNDER_NEW)
	
	return L_DO_DW_NEW, L_DO_DB_NEW, L_DO_DWUNDER_NEW, L_DO_DBUNDER_NEW

#########
# 20.8% of reverse_pass_partials()
def do_dw__mem_prev(L_W, L_DO_DW, L_DO_DB, L_DO_DWUNDER,L_DO_DBUNDER, L_O, MEM_PREV, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, DO_DO_IN, O_PREV):
	DO_CONTENT_DMEM_PREV = init_buffer()
	DO_DMEM_PREV = init_buffer()
	
	DO_DO_CONTENT = do_do_content__(L_O, DO_DO_IN, O_PREV) # 17.9%
	MEM_PREV[1] = (6,8)
	nm.cosine_sim_expand_dmem(L_O[KEY], MEM_PREV, DO_CONTENT_DMEM_PREV) # 15.5%
	#print DO_DO_CONTENT[1], DO_CONTENT_DMEM_PREV[1], L_O[CONTENT][1], DO_DMEM_PREV[1]
	DO_DO_CONTENT[1] = (L_O[CONTENT][1][0], L_O[CONTENT][1][1], L_O[CONTENT][1][0], L_O[CONTENT][1][1])
	#print DO_DO_CONTENT[1], DO_CONTENT_DMEM_PREV[1], L_O[CONTENT][1], DO_DMEM_PREV[1]
	nm.mult_partials(DO_DO_CONTENT, DO_CONTENT_DMEM_PREV, L_O[CONTENT], DO_DMEM_PREV)
	
	L_DO_DW_NEW = set_list_buffer(return_list_buffer(L_DO_DW))
	L_DO_DB_NEW = set_list_buffer(return_list_buffer(L_DO_DB))
	L_DO_DWUNDER_NEW = set_list_buffer(return_list_buffer(L_DO_DWUNDER))
	L_DO_DBUNDER_NEW = set_list_buffer(return_list_buffer(L_DO_DBUNDER))
	
	if len(L_DMEM_PREV_DWW[0][1]) == 4:
		DO_DMEM_PREV[1] = (16,6, 6, 8)
	else:
		DO_DMEM_PREV[1] = (16,6, 48)
		MEM_PREV[1] = (48,)
	nm.mult_partials__layers(DO_DMEM_PREV, L_DMEM_PREV_DWW, MEM_PREV, L_DO_DW_NEW, increment=1) # 49.6%
	nm.mult_partials__layers(DO_DMEM_PREV, L_DMEM_PREV_DBW, MEM_PREV, L_DO_DB_NEW, increment=1)
	nm.mult_partials__layers(DO_DMEM_PREV, L_DMEM_PREV_DWUNDER, MEM_PREV, L_DO_DWUNDER_NEW, increment=1)
	nm.mult_partials__layers(DO_DMEM_PREV, L_DMEM_PREV_DBUNDER, MEM_PREV, L_DO_DBUNDER_NEW, increment=1)
	
	return L_DO_DW_NEW, L_DO_DB_NEW, L_DO_DWUNDER_NEW, L_DO_DBUNDER_NEW

	
def mem_partials_gpu(L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER,L_DMEM_PREV_DBUNDER, L_DOW_DWW, L_DOW_DBW,L_DOW_DWUNDER,L_DOW_DBUNDER,L_OW_PREV, L_OUNDER_PREV, L_WW,L_BW, L_WUNDER, L_BUNDER, X_PREV, MEM_PREV_PREV):
	
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
	MEM_PREV_TIMES_DE_DOW[1] = (6, 8, 16*6)
	L_OW_PREV[F][1] = (16*6,)
	nm.mult_partials__layers(MEM_PREV_TIMES_DE_DOW, L_DOW_DWW, L_OW_PREV[F], L_DMEM_PREV_DWW)
	nm.mult_partials__layers(MEM_PREV_TIMES_DE_DOW, L_DOW_DBW, L_OW_PREV[F], L_DMEM_PREV_DBW)
	nm.mult_partials__layers(MEM_PREV_TIMES_DE_DOW, L_DOW_DWUNDER, L_OW_PREV[F], L_DMEM_PREV_DWUNDER)
	nm.mult_partials__layers(MEM_PREV_TIMES_DE_DOW, L_DOW_DBUNDER, L_OW_PREV[F], L_DMEM_PREV_DBUNDER)
	
	# W[ERASE] gradients (de wrt W[ERASE])
	L_OW_PREV[F][1] = (16,6)
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
	DA_DOW[1] = (6, 8, 16*6)
	L_OW_PREV[F][1] = (16*6,)
	nm.mult_partials__layers(DA_DOW, L_DOW_DWW, L_OW_PREV[F], L_DMEM_PREV_DWW, increment=1) # da_dlayer
	nm.mult_partials__layers(DA_DOW, L_DOW_DBW, L_OW_PREV[F], L_DMEM_PREV_DBW, increment=1) # da_dlayer
	nm.mult_partials__layers(DA_DOW, L_DOW_DWUNDER, L_OW_PREV[F], L_DMEM_PREV_DWUNDER, increment=1)
	nm.mult_partials__layers(DA_DOW, L_DOW_DBUNDER, L_OW_PREV[F], L_DMEM_PREV_DBUNDER, increment=1)
	
	###
	# W[ADD] gradients
	L_OW_PREV[F][1] = (16,6)
	nm.add_mem_dadd_out(L_OW_PREV[F], L_OW_PREV[ADD], DA_DADD_OUT)
	
	nm.point_wise_add(L_DMEM_PREV_DBW[ADD], DA_DADD_OUT)
	nm.linear_2d_F_dF(L_WW[ADD], L_OUNDER_PREV[F_UNDER], DADD_OUT_DWADD)
	nm.mult_partials(DA_DADD_OUT, DADD_OUT_DWADD, L_OW_PREV[ADD], L_DMEM_PREV_DWW[ADD], increment=1) # da_dwadd
	
	# under: (wrt inputs)
	nm.linear_2d_F_dx(L_WW[ADD], L_OUNDER_PREV[F_UNDER], DADD_OUT_DG3UNDER)
	nm.mult_partials(DA_DADD_OUT, DADD_OUT_DG3UNDER, L_OW_PREV[ADD], DA_DG3UNDER) # da_dwunder
	
	nm.mult_partials__layers(DA_DG3UNDER, DG3UNDER_DW, L_OUNDER_PREV[F_UNDER], L_DMEM_PREV_DWUNDER, increment=1, squeeze=1)
	nm.mult_partials__layers(DA_DG3UNDER, DG3UNDER_DB, L_OUNDER_PREV[F_UNDER], L_DMEM_PREV_DBUNDER, increment=1, squeeze=1)
	
	return L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER

### compute state partials (stores history, in a sense) 
# 74.7% of main()
#@profile
def reverse_pass_partials(L_WUNDER,L_BUNDER, L_WR,L_WW,L_BR,L_BW, L_OUNDER, L_OUNDER_PREV, L_OR, L_OR_PREV, L_OW_PREV, L_OW_PREV_PREV, MEM_PREV, MEM_PREV_PREV, X, X_PREV, frame, L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, L_DOR_DWR, L_DOR_DBR, L_DOR_DWUNDER, L_DOR_DBUNDER, L_DOR_DWW, L_DOR_DBW):
	DOR_DGSHARPEN = init_buffer()
	DOW_PREV_DGSHARPEN = init_buffer()
	DGSHARPEN_DOR_IN = init_buffer()
	DGSHARPEN_DOW_PREV_IN = init_buffer()
	DOR_DOR_IN = init_buffer()
	DOW_PREV_DOW_PREV_IN = init_buffer()
	
	nm.sharpen_dw(L_OR[SHIFTED], L_OR[GAMMA], DOR_DGSHARPEN)
	nm.sharpen_dw(L_OW_PREV[SHIFTED], L_OW_PREV[GAMMA], DOW_PREV_DGSHARPEN)
	
	nm.shift_w_dw_interp(L_OR[SHIFT], L_OR[IN], DGSHARPEN_DOR_IN)
	nm.shift_w_dw_interp(L_OW_PREV[SHIFT], L_OW_PREV[IN], DGSHARPEN_DOW_PREV_IN)
	
	nm.mult_partials(DOR_DGSHARPEN, DGSHARPEN_DOR_IN, L_OR[SHARPENED], DOR_DOR_IN)
	nm.mult_partials(DOW_PREV_DGSHARPEN, DGSHARPEN_DOW_PREV_IN, L_OW_PREV[SHARPENED], DOW_PREV_DOW_PREV_IN)
	
	# partials for write head output (OW)
	if frame > 1:
		L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER = do_dw__o_prev(L_WW, L_OW_PREV_PREV[F], L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER, L_OW_PREV, DOW_PREV_DOW_PREV_IN) # 13.4%
		L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER = do_dw__mem_prev(L_WW, L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER, L_OW_PREV, MEM_PREV_PREV, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER,  DOW_PREV_DOW_PREV_IN, L_OW_PREV_PREV[F]) # 10.3%
		L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER = do_dw__inputs_gpu(L_WW, L_WUNDER, L_BUNDER, L_OW_PREV_PREV[F], L_OUNDER_PREV, L_DOW_DWUNDER, L_DOW_DBUNDER, L_OW_PREV, L_DOW_DWW, L_DOW_DBW, MEM_PREV_PREV, X_PREV, DOW_PREV_DOW_PREV_IN) # 12.6%
		
		DMEM_PREV_DWW, DMEM_PREV_DBW, DMEM_PREV_DWUNDER, DMEM_PREV_DBUNDER = mem_partials_gpu(L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER, L_OW_PREV, L_OUNDER_PREV, L_WW, L_BW, L_WUNDER, L_BUNDER, X_PREV, MEM_PREV_PREV) # 18.6%
	
	# partials from read head output (OR)
	L_DOR_DWR, L_DOR_DBR, L_DOR_DWUNDER, L_DOR_DBUNDER = do_dw__o_prev(L_WR, L_OR_PREV[F], L_DOR_DWR, L_DOR_DBR, L_DOR_DWUNDER, L_DOR_DBUNDER, L_OR, DOR_DOR_IN) # 7.8%
	L_DOR_DWR, L_DOR_DBR, L_DOR_DWUNDER, L_DOR_DBUNDER = do_dw__inputs_gpu(L_WR, L_WUNDER, L_BUNDER, L_OR_PREV[F], L_OUNDER, L_DOR_DWUNDER, L_DOR_DBUNDER, L_OR, L_DOR_DWR, L_DOR_DBR, MEM_PREV, X, DOR_DOR_IN) # 12.3%
	
	L_DOR_DWW, L_DOR_DBW = do_dw__o_prev(L_WR, L_OR_PREV[F], L_DOR_DWW, L_DOR_DBW, L_DOR_DWUNDER, L_DOR_DBUNDER, L_OR, DOR_DOR_IN)[:2] #?... 13.4%
	L_DOR_DWW, L_DOR_DBW, L_DOR_DWUNDER, L_DOR_DBUNDER = do_dw__mem_prev(L_WR, L_DOR_DWW, L_DOR_DBW, L_DOR_DWUNDER, L_DOR_DBUNDER, L_OR, MEM_PREV, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, DOR_DOR_IN, L_OR_PREV[F]) # 10.4%
	
	return L_DOW_DWW, L_DOW_DBW, L_DOW_DWUNDER, L_DOW_DBUNDER, L_DMEM_PREV_DWW, L_DMEM_PREV_DBW, L_DMEM_PREV_DWUNDER, L_DMEM_PREV_DBUNDER, L_DOR_DWR, L_DOR_DBR, L_DOR_DWUNDER, L_DOR_DBUNDER, L_DOR_DWW, L_DOR_DBW



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
	MEM_PREV[1] = (6,8)
	nm.linear_F_dF(L_OR[F], MEM_PREV, DREAD_MEM_DOR)
	nm.mult_partials(DERR_DREAD_MEM, DREAD_MEM_DOR, READ_MEM, DERR_DOR)
	
	L_OR[F][1] = (16*6,)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DWR, L_OR[F], L_DWR)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DBR, L_OR[F], L_DBR)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DWW, L_OR[F], L_DWW)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DBW, L_OR[F], L_DBW)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DWUNDER, L_OR[F], L_DWUNDER)
	nm.mult_partials__layers(DERR_DOR, L_DOR_DBUNDER, L_OR[F], L_DBUNDER)
	
	# write weights
	L_OR[F][1] = (16,6)
	nm.linear_F_dx(L_OR[F], MEM_PREV, DREAD_MEM_DMEM_PREV)
	nm.mult_partials(DERR_DREAD_MEM, DREAD_MEM_DMEM_PREV, READ_MEM, DERR_DMEM_PREV)
	
	MEM_PREV[1] = (6*8,)
	nm.mult_partials__layers(DERR_DMEM_PREV, L_DMEM_PREV_DWW, MEM_PREV, L_DWW, increment=1)
	nm.mult_partials__layers(DERR_DMEM_PREV, L_DMEM_PREV_DBW, MEM_PREV, L_DBW, increment=1)
	nm.mult_partials__layers(DERR_DMEM_PREV, L_DMEM_PREV_DWUNDER, MEM_PREV, L_DWUNDER, increment=1)
	nm.mult_partials__layers(DERR_DMEM_PREV, L_DMEM_PREV_DBUNDER, MEM_PREV, L_DBUNDER, increment=1)
	
	return L_DWR, L_DBR, L_DWW, L_DBW, L_DWUNDER, L_DBUNDER, L_DWABOVE, L_DBABOVE

