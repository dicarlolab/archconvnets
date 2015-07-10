import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *
from init_vars import *

########
def weight_address(W, o_prev, x_cur, shift_out, o_content): # todo: shift_out, o_content computations
	G = [None]*4
	O = [None]*3
	
	G[SHIFT] = sq_F(W[SHIFT], x_cur)
	
	G[L1] = sq_F(W[L1], x_cur)
	G[L2] = sq_F(W[L2], G[L1])
	G[L3] = sq_F(W[L3], G[L2])
	O[IN] = interpolate_simp(o_prev, G[L3]) + interpolate_simp(o_content, G[L3])
	O[SQ] = sq_points(O[IN])
	
	O[F] = shift_w(shift_out, O[SQ])
	
	return G, O

def forward_pass(W,WW, o_prev, ow_prev, mem_prev,x_cur):
	G, O = weight_address(W, o_prev, x_cur, shift_out, o_content)
	GW, OW = weight_address(WW, ow_prev, x_cur, shiftw_out, ow_content)
	
	read_mem = linear_F(O[F], mem_prev)
	mem = mem_prev + add_mem(OW[F], add_out)
	
	return O,OW,mem,read_mem,G,GW

##########
def weight_address_partials(W, o_prev, shift_out, o_content, x_cur, DO_DW, DO_CONTENT_DW, G,O):
	DO_DW_NEW = copy.deepcopy(DO_DW)
	
	do_do_sq = shift_w_dw_interp_nsum(shift_out)
	do_sq_do_in = sq_points_dinput(O[IN])
	do_do_in = mult_partials(do_do_sq, do_sq_do_in, O[SQ])
	
	#w3
	dg3_dg2 = sq_dlayer_in_nsum(W[L3], G[L2])
	dg3_dw3 = sq_dF_nsum(W[L3], G[L2], G[L3])
	
	# w2
	dg2_dg1 = sq_dlayer_in_nsum(W[L2], G[L1])
	dg2_dw2 = sq_dF_nsum(W[L2], G[L1], G[L2])
	dg3_dw2 = mult_partials(dg3_dg2, dg2_dw2, np.squeeze(G[L2]))
	
	# w1:
	dg1_dw1 = sq_dF_nsum(W[L1], x_cur, G[L1])
	dg3_dg1 = mult_partials(dg3_dg2, dg2_dg1, np.squeeze(G[L2]))
	dg3_dw1 = mult_partials(dg3_dg1, dg1_dw1, np.squeeze(G[L1]))
	
	
	DO_DW_NEW[L3] = interpolate_simp_dx(dg3_dw3, DO_DW[L3], DO_CONTENT_DW[L3], G[L3], o_prev, o_content, do_do_in)
	DO_DW_NEW[L2] = interpolate_simp_dx(dg3_dw2, DO_DW[L2], DO_CONTENT_DW[L2], G[L3], o_prev, o_content, do_do_in)
	DO_DW_NEW[L1] = interpolate_simp_dx(dg3_dw1, DO_DW[L1], DO_CONTENT_DW[L1], G[L3], o_prev, o_content, do_do_in)
	
	return DO_DW_NEW

def mem_partials(add_out, DMEM_PREV_DWW, DOW_DWW, OW_PREV):
	DMEM_PREV_DWW_NEW = copy.deepcopy(DMEM_PREV_DWW)
	
	da_dow = add_mem_dgw(add_out)
	
	da_dww3 = mult_partials(da_dow, DOW_DWW[L3], OW_PREV[F])
	da_dww2 = mult_partials(da_dow, DOW_DWW[L2], OW_PREV[F])
	da_dww1 = mult_partials(da_dow, DOW_DWW[L1], OW_PREV[F])
	
	DMEM_PREV_DWW_NEW[L3] = DMEM_PREV_DWW[L3] + da_dww3
	DMEM_PREV_DWW_NEW[L2] = DMEM_PREV_DWW[L2] + da_dww2
	DMEM_PREV_DWW_NEW[L1] = DMEM_PREV_DWW[L1] + da_dww1
	
	return DMEM_PREV_DWW_NEW

#####
DERIV_L = L2

def f(y):
	W[DERIV_L][i_ind,j_ind] = y
	
	O_PREV = copy.deepcopy(O_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	mem_prev = copy.deepcopy(mem_previ); mem = np.zeros_like(mem_prev)
	
	for frame in range(1,N_FRAMES+1):
		O_PREV, OW_PREV, mem_prev, read_mem = forward_pass(W, WW, O_PREV[F], OW_PREV[F], mem_prev, x[frame])[:4]
	
	return ((read_mem - t)**2).sum()


def g(y):
	W[DERIV_L][i_ind,j_ind] = y
	
	GW_PREV = copy.deepcopy(GW_PREVi)
	O_PREV = copy.deepcopy(O_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	OW_PREV_PREV = copy.deepcopy(OW_PREV_PREVi)
	DMEM_PREV_DWW = copy.deepcopy(DMEM_PREV_DWWi)
	DO_DW = copy.deepcopy(DO_DWi); DOW_DWW = copy.deepcopy(DO_DWWi)
	mem_prev = copy.deepcopy(mem_previ); mem = np.zeros_like(mem_prev); 
	
	for frame in range(1,N_FRAMES+1):
		# forward
		O,OW,mem,read_mem,G,GW = forward_pass(W, WW, O_PREV[F], OW_PREV[F], mem_prev, x[frame])
		
		# partials for weight addresses
		DO_DW = weight_address_partials(W,O_PREV[F], shift_out, o_content, x[frame], DO_DW, DO_CONTENT_DW, G,O)
		DOW_DWW = weight_address_partials(WW, OW_PREV_PREV[F], shiftw_out, ow_content, x[frame-1], DOW_DWW, DO_CONTENT_DW, GW_PREV, OW_PREV)
		
		# partials for mem
		DMEM_PREV_DWW = mem_partials(add_out, DMEM_PREV_DWW, DOW_DWW, OW_PREV)
		
		# update temporal state vars
		if frame != N_FRAMES:
			OW_PREV_PREV = copy.deepcopy(OW_PREV)
			OW_PREV = copy.deepcopy(OW); O_PREV = copy.deepcopy(O)
			mem_prev = copy.deepcopy(mem)
			GW_PREV = copy.deepcopy(GW)
	
	########
	## full gradients:
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	# read weights
	dread_mem_do = linear_F_dF_nsum(mem_prev)
	derr_do = mult_partials(derr_dread_mem, dread_mem_do, read_mem)
	
	DW[L1] = mult_partials_sum(derr_do, DO_DW[L1], O[F])
	DW[L2] = mult_partials_sum(derr_do, DO_DW[L2], O[F])
	DW[L3] = mult_partials_sum(derr_do, DO_DW[L3], O[F])

	# write weights
	dread_mem_dmem_prev = linear_F_dx_nsum(O[F])
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	DWW[L3] = mult_partials_sum(derr_dmem_prev, DMEM_PREV_DWW[L3], mem_prev)
	DWW[L2] = mult_partials_sum(derr_dmem_prev, DMEM_PREV_DWW[L2], mem_prev)
	DWW[L1] = mult_partials_sum(derr_dmem_prev, DMEM_PREV_DWW[L1], mem_prev)
	
	return DW[DERIV_L][i_ind,j_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = W[DERIV_L]
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()
