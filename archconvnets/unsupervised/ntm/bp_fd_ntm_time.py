#from archconvnets.unsupervised.cudnn_module.cudnn_module import *
import time
import numpy as np
from scipy.io import savemat, loadmat
import copy
from scipy.stats import zscore
import random
import scipy
from ntm_gradients import *

n_shifts = 3
C = 4
M = 5
mem_length = 8
n2 = 6
n1 = 7
n_in = 3

SCALE = .6
N_FRAMES = 3

mem_previ = np.random.normal(size=(M, mem_length))

o_previ = np.random.normal(size=(C,M))
ow_previ = np.random.normal(size=(C,M))
o_content = np.random.normal(size=(C,M))
ow_content = np.random.normal(size=(C,M))

w3 = np.random.normal(size=(C,n2)) * SCALE
w2 = np.random.normal(size=(n2,n1)) * SCALE
w1 = np.random.normal(size=(n1,n_in)) * SCALE

ww1 = np.random.normal(size=(n1, n_in)) * SCALE
ww2 = np.random.normal(size=(n2, n1)) * SCALE
ww3 = np.random.normal(size=(C, n2)) * SCALE

W = [w1, w2, w3]
WW = [ww1, ww2, ww3]


shift_out = np.random.normal(size=(C, n_shifts))
shiftw_out = np.random.normal(size=(C, n_shifts))
add_out = np.random.normal(size=(C, mem_length)) * SCALE

x = np.random.normal(size=(N_FRAMES+1, n_in,1)) * SCALE
t = np.random.normal(size=(C,mem_length))

x[0] = np.zeros_like(x[0])

do_dw3 = np.zeros((C,M,C,n2))
do_dw2 = np.zeros((C,M,n2,n1))
do_dw1 = np.zeros((C,M,n1,n_in))

dow_dww1 = np.zeros((C,M, n1,n_in))
dow_dww2 = np.zeros((C,M, n2,n1))
dow_dww3 = np.zeros((C,M, C,n2))

DO_DWi = [do_dw1, do_dw2, do_dw3]
DO_DWWi = [dow_dww1, dow_dww2, dow_dww3]

do_content_dw3 = np.zeros_like(do_dw3)
do_content_dw2 = np.zeros_like(do_dw2)
do_content_dw1 = np.zeros_like(do_dw1)

dow_content_dww1 = np.zeros_like(dow_dww1)
dow_content_dww2 = np.zeros_like(dow_dww2)
dow_content_dww3 = np.zeros_like(dow_dww3)

DO_CONTENT_DW = [do_content_dw1, do_content_dw2, do_content_dw3]
DOW_CONTENT_DWW = [dow_content_dww1, dow_content_dww2, dow_content_dww3]

dmem_prev_dww1i = np.zeros((M, mem_length, n1,n_in))
dmem_prev_dww2i = np.zeros((M, mem_length, n2,n1))
dmem_prev_dww3i = np.zeros((M, mem_length, C,n2))

############
def linear_F_dx_nsum(o):
	n = mem_previ.shape[1]
	temp = np.zeros((o_previ.shape[0], n, mem_previ.shape[0], n))
	temp[:,range(n),:,range(n)] = o
	return temp

def linear_F_dF_nsum(mem):
	n = o_previ.shape[0]
	temp = np.zeros((n, mem.shape[1], n, o_previ.shape[1]))
	temp[range(n),:,range(n)] = mem.T
	return temp

################
def shift_w_dw_interp_nsum(shift_out):
	# shift_out: [n_controllers, n_shifts]
	temp = np.zeros((C, M, C, M))
	
	for loc in range(M):
		temp[range(C),loc,range(C),loc-1] = shift_out[:,0]
		temp[range(C),loc,range(C),loc] = shift_out[:,1]
		temp[range(C),loc,range(C),(loc+1)%M] = shift_out[:,2]
			
	return temp # [n_controllers, M, n_controllers, M]

#####
def add_mem(gw, add_out):
	return np.dot(gw.T, add_out)

def add_mem_dgw(add_out):
	temp = np.zeros((M, mem_length, C, M))
	temp[range(M),:,:,range(M)] = add_out.T
	return temp

################# interpolate simplified
def interpolate_simp(w_prev, interp_gate_out):
	return w_prev * interp_gate_out

def interpolate_simp_dx_nprod(dg3_dx, do_dx, do_content_dx, g3, o_prev, o_content):
	do_in_dx = np.einsum(do_dx + do_content_dx, range(4), g3, [0,1], range(4))
	do_in_dx += np.einsum(o_prev + o_content, [0,1], dg3_dx, [0,2,3], range(4))
	
	return do_in_dx
	
def interpolate_simp_dx(dg3_dx, do_dx, do_content_dx, g3, o_prev, o_content, do_do_in):
	do_in_dx = interpolate_simp_dx_nprod(dg3_dx, do_dx, do_content_dx, g3, o_prev, o_content)
	
	do_dx = mult_partials(do_do_in, do_in_dx, o_prev)
	return do_dx

########
L1 = 0; L2 = 1; L3 = 2
IN = 0; SQ = 1; F = 2
O_PREVi = [None, None, o_previ]
OW_PREVi = [np.zeros_like(ow_previ), np.zeros_like(ow_previ), ow_previ]
OW_PREV_PREVi = [None, None, np.zeros_like(ow_previ)]
GW_PREVi = [np.zeros((n1,1)), np.zeros((n2,1)), np.zeros((C,1))]

def weight_address(W, o_prev, x_cur, shift_out, o_content): # todo: shift_out, o_content computations
	G = [None]*3
	O = [None]*3
	
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

def f(y):
	WW[L1][i_ind,j_ind] = y
	
	O_PREV = copy.deepcopy(O_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	mem_prev = copy.deepcopy(mem_previ); mem = np.zeros_like(mem_prev)
	
	for frame in range(1,N_FRAMES+1):
		O_PREV, OW_PREV, mem_prev, read_mem = forward_pass(W, WW, O_PREV[F], OW_PREV[F], mem_prev, x[frame])[:4]
	
	return ((read_mem - t)**2).sum()


def g(y):
	WW[L1][i_ind,j_ind] = y
	
	GW_PREV = copy.deepcopy(GW_PREVi)
	O_PREV = copy.deepcopy(O_PREVi); OW_PREV = copy.deepcopy(OW_PREVi)
	OW_PREV_PREV = copy.deepcopy(OW_PREV_PREVi)
	
	DO_DW = copy.deepcopy(DO_DWi); DOW_DWW = copy.deepcopy(DO_DWWi)
	
	dmem_prev_dww3 = copy.deepcopy(dmem_prev_dww3i); 
	dmem_prev_dww2 = copy.deepcopy(dmem_prev_dww2i); 
	dmem_prev_dww1 = copy.deepcopy(dmem_prev_dww1i); 
	mem_prev = copy.deepcopy(mem_previ); mem = np.zeros_like(mem_prev); 
	
	
	for frame in range(1,N_FRAMES+1):
		# forward
		O,OW,mem,read_mem,G,GW = forward_pass(W, WW, O_PREV[F], OW_PREV[F], mem_prev, x[frame])
		
		# partials for weight addresses
		DO_DW = weight_address_partials(W,O_PREV[F], shift_out, o_content, x[frame], DO_DW, DO_CONTENT_DW, G,O)
		DOW_DWW = weight_address_partials(WW, OW_PREV_PREV[F], shiftw_out, ow_content, x[frame-1], DOW_DWW, DO_CONTENT_DW, GW_PREV, OW_PREV)

		
		# partials for mem
		da_dow = add_mem_dgw(add_out)
		
		da_dww3 = mult_partials(da_dow, DOW_DWW[L3], OW_PREV[F])
		da_dww2 = mult_partials(da_dow, DOW_DWW[L2], OW_PREV[F])
		da_dww1 = mult_partials(da_dow, DOW_DWW[L1], OW_PREV[F])
		
		dmem_prev_dww3 += da_dww3
		dmem_prev_dww2 += da_dww2
		dmem_prev_dww1 += da_dww1
		
		# update temporal state vars
		if frame != N_FRAMES:
			OW_PREV_PREV = copy.deepcopy(OW_PREV)
			OW_PREV = copy.deepcopy(OW); O_PREV = copy.deepcopy(O)
			mem_prev = copy.deepcopy(mem)
			GW_PREV = copy.deepcopy(GW)
	
	# full gradients:
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	dread_mem_do = linear_F_dF_nsum(mem_prev)
	dread_mem_dmem_prev = linear_F_dx_nsum(O[F])

	derr_do = mult_partials(derr_dread_mem, dread_mem_do, read_mem)
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	dww3 = mult_partials_sum(derr_dmem_prev, dmem_prev_dww3, mem_prev)
	dww2 = mult_partials_sum(derr_dmem_prev, dmem_prev_dww2, mem_prev)
	dww1 = mult_partials_sum(derr_dmem_prev, dmem_prev_dww1, mem_prev)
	
	dw1 = mult_partials_sum(derr_do, DO_DW[L1], O[F])
	dw2 = mult_partials_sum(derr_do, DO_DW[L2], O[F])
	dw3 = mult_partials_sum(derr_do, DO_DW[L3], O[F])
	
	return dww1[i_ind,j_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = ww1
	#ref = W[L3]
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()


