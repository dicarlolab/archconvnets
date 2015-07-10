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

W = [w1, w2, w3]

ww2 = np.random.normal(size=(n2, n_in)) * SCALE
ww3 = np.random.normal(size=(C, n2)) * SCALE

shift_out = np.random.normal(size=(C, n_shifts))
shiftw_out = np.random.normal(size=(C, n_shifts))
add_out = np.random.normal(size=(C, mem_length)) * SCALE

x = np.random.normal(size=(N_FRAMES+1, n_in,1)) * SCALE
t = np.random.normal(size=(C,mem_length))

x[0] = np.zeros_like(x[0])

do_dw3 = np.zeros((C,M,C,n2))
do_dw2 = np.zeros((C,M,n2,n1))
do_dw1 = np.zeros((C,M,n1,n_in))

DO_DW = [do_dw1, do_dw2, do_dw3]

dow_dww2i = np.zeros((C,M, n2,n_in))
dow_dww3i = np.zeros((C,M, C,n2))

do_content_dw3 = np.zeros_like(do_dw3)
do_content_dw2 = np.zeros_like(do_dw2)
do_content_dw1 = np.zeros_like(do_dw1)

dow_content_dww2 = np.zeros_like(dow_dww2i)
dow_content_dww3 = np.zeros_like(dow_dww3i)

dmem_prev_dww2i = np.zeros((M, mem_length, n2,n_in))
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
Oi = [None, None, o_previ]

def forward_pass(W,ww2, ww3, o_prev, ow_prev, mem_prev,x_cur):
	G = [None]*3
	O = [None]*3
	
	G[L1] = sq_F(W[L1], x_cur)
	G[L2] = sq_F(W[L2], G[L1])
	G[L3] = sq_F(W[L3], G[L2])
	O[IN] = interpolate_simp(o_prev, G[L3]) + interpolate_simp(o_content, G[L3])
	O[SQ] = sq_points(O[IN])
	O[F] = shift_w(shift_out, O[SQ])
	read_mem = linear_F(O[F], mem_prev)
	
	gw2 = sq_F(ww2,x_cur)
	gw3 = sq_F(ww3,gw2)
	ow_in = interpolate_simp(ow_prev, gw3) + interpolate_simp(ow_content, gw3)
	ow_sq = sq_points(ow_in)
	ow = shift_w(shiftw_out, ow_sq)
	mem = mem_prev + add_mem(ow, add_out)
	
	return O,ow,mem,read_mem,G,gw2,gw3,ow_in,ow_sq

##########
def compute_weight_address_partials(W, o_prev, o_content, x_cur, DO_DW, G,O):
	## read gradients
	do_do_sq = shift_w_dw_interp_nsum(shift_out)
	do_sq_do_in = sq_points_dinput(O[IN])
	do_do_in = mult_partials(do_do_sq, do_sq_do_in, O[SQ])
	
	#w3
	dg3_dg2 = sq_dlayer_in_nsum(W[L3], G[L2])
	dg3_dw3 = sq_dF_nsum(W[L3], G[L2], G[L3])
	DO_DW[L3] = interpolate_simp_dx(dg3_dw3, DO_DW[L3], do_content_dw3, G[L3], o_prev, o_content, do_do_in)
	
	# w2
	dg2_dg1 = sq_dlayer_in_nsum(W[L2], G[L1])
	dg2_dw2 = sq_dF_nsum(W[L2], G[L1], G[L2])
	dg3_dw2 = mult_partials(dg3_dg2, dg2_dw2, np.squeeze(G[L2]))
	DO_DW[L2] = interpolate_simp_dx(dg3_dw2, DO_DW[L2], do_content_dw2, G[L3], o_prev, o_content, do_do_in)
	
	# w1:
	dg1_dw1 = sq_dF_nsum(w1, x_cur, G[L1])
	dg3_dg1 = mult_partials(dg3_dg2, dg2_dg1, np.squeeze(G[L2]))
	dg3_dw1 = mult_partials(dg3_dg1, dg1_dw1, np.squeeze(G[L1]))
	DO_DW[L1] = interpolate_simp_dx(dg3_dw1, DO_DW[L1], do_content_dw1, G[L3], o_prev, o_content, do_do_in)
	
	return DO_DW

def f(y):
	#ww2[i_ind,j_ind] = y
	W[L3][i_ind,j_ind] = y
	
	O = copy.deepcopy(Oi)
	
	ow_prev = copy.deepcopy(ow_previ)
	mem_prev = copy.deepcopy(mem_previ); mem = np.zeros_like(mem_prev)
	
	for frame in range(1,N_FRAMES+1):
		O, ow_prev, mem_prev, read_mem = forward_pass(W,ww2, ww3, O[F], ow_prev, mem_prev,x[frame])[:4]
	
	return ((read_mem - t)**2).sum()


def g(y):
	#ww2[i_ind,j_ind] = y
	W[L3][i_ind,j_ind] = y
	
	O = copy.deepcopy(Oi)
	O_PREV = copy.deepcopy(O)
	
	DO_DW = [do_dw1, do_dw2, do_dw3]
	
	dow_dww3 = copy.deepcopy(dow_dww3i)
	dow_dww2 = copy.deepcopy(dow_dww2i)
	o_prev = copy.deepcopy(o_previ); ow_prev = copy.deepcopy(ow_previ)
	ow_prev_prev = np.zeros_like(ow_prev);
	ow_in_prev = np.zeros_like(ow_prev);
	ow_sq_prev = np.zeros_like(ow_prev);
	dmem_prev_dww3 = copy.deepcopy(dmem_prev_dww3i); 
	dmem_prev_dww2 = copy.deepcopy(dmem_prev_dww2i); 
	mem_prev = copy.deepcopy(mem_previ); mem = np.zeros_like(mem_prev); 
	gw2_prev = np.zeros((n2,1)); gw3_prev = np.zeros((C,1))
	
	for frame in range(1,N_FRAMES+1):
		# forward
		O,ow,mem,read_mem,G,gw2,gw3,ow_in,ow_sq = forward_pass(W, ww2, ww3, O_PREV[F], ow_prev, mem_prev,x[frame])
		
		# partials for previous address weights (o)
		DO_DW = compute_weight_address_partials(W,O_PREV[F], o_content, x[frame], DO_DW,G,O)
		
		########### temp
		## write partials (for the previous time step---read then write!)
		dow_dow_sq = shift_w_dw_interp_nsum(shiftw_out)
		dow_sq_dow_in = sq_points_dinput(ow_in_prev)
		dow_dow_in = mult_partials(dow_dow_sq, dow_sq_dow_in, ow_sq_prev)
		
		# ww3:
		dgw3_dgw2 = sq_dlayer_in_nsum(ww3, gw2_prev)
		dgw3_dww3 = sq_dF_nsum(ww3, gw2_prev, gw3_prev)
		dow_dww3 = interpolate_simp_dx(dgw3_dww3, dow_dww3, dow_content_dww3, gw3_prev, ow_prev_prev, ow_content,dow_dow_in)
		
		# w2
		#dgw2_dgw1 = sq_dlayer_in_nsum(ww2, gw1)
		dgw2_dww2 = sq_dF_nsum(ww2, x[frame-1], gw2_prev)
		dgw3_dww2 = mult_partials(dgw3_dgw2, dgw2_dww2, np.squeeze(gw2_prev))
		dow_dww2 = interpolate_simp_dx(dgw3_dww2, dow_dww2, dow_content_dww2, gw3_prev, ow_prev_prev, ow_content, dow_dow_in)
		
		'''##############
		#w3
		dg3_dg2 = sq_dlayer_in_nsum(W[L3], G[L2])
		dg3_dw3 = sq_dF_nsum(W[L3], G[L2], G[L3])
		DO_DW[L3] = interpolate_simp_dx(dg3_dw3, DO_DW[L3], do_content_dw3, G[L3], o_prev, o_content, do_do_in)
		
		# w2
		dg2_dg1 = sq_dlayer_in_nsum(W[L2], G[L1])
		dg2_dw2 = sq_dF_nsum(W[L2], G[L1], G[L2])
		dg3_dw2 = mult_partials(dg3_dg2, dg2_dw2, np.squeeze(G[L2]))
		DO_DW[L2] = interpolate_simp_dx(dg3_dw2, DO_DW[L2], do_content_dw2, G[L3], o_prev, o_content, do_do_in)
		
		# w1:
		dg1_dw1 = sq_dF_nsum(w1, x_cur, G[L1])
		dg3_dg1 = mult_partials(dg3_dg2, dg2_dg1, np.squeeze(G[L2]))
		dg3_dw1 = mult_partials(dg3_dg1, dg1_dw1, np.squeeze(G[L1]))
		DO_DW[L1] = interpolate_simp_dx(dg3_dw1, DO_DW[L1], do_content_dw1, G[L3], o_prev, o_content, do_do_in)'''
		
		###########################################
		
		
		# partials for mem
		da_dow = add_mem_dgw(add_out)
		
		da_dww3 = mult_partials(da_dow, dow_dww3, ow_prev)
		da_dww2 = mult_partials(da_dow, dow_dww2, ow_prev)
		
		dmem_prev_dww3 += da_dww3
		dmem_prev_dww2 += da_dww2
		
		# update temporal state vars
		if frame != N_FRAMES:
			ow_prev_prev = copy.deepcopy(ow_prev)
			O_PREV = copy.deepcopy(O); mem_prev = copy.deepcopy(mem)
			ow_prev = copy.deepcopy(ow); gw3_prev = copy.deepcopy(gw3); gw2_prev = copy.deepcopy(gw2)
			ow_in_prev = copy.deepcopy(ow_in); ow_sq_prev = copy.deepcopy(ow_sq)
	
	# full gradients:
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	dread_mem_do = linear_F_dF_nsum(mem_prev)
	dread_mem_dmem_prev = linear_F_dx_nsum(O[F])

	derr_do = mult_partials(derr_dread_mem, dread_mem_do, read_mem)
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	dww3 = mult_partials_sum(derr_dmem_prev, dmem_prev_dww3, mem_prev)
	dww2 = mult_partials_sum(derr_dmem_prev, dmem_prev_dww2, mem_prev)
	
	dw1 = mult_partials_sum(derr_do, DO_DW[L1], O[F])
	dw2 = mult_partials_sum(derr_do, DO_DW[L2], O[F])
	dw3 = mult_partials_sum(derr_do, DO_DW[L3], O[F])
	
	return dw3[i_ind,j_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	#ref = ww2
	ref = W[L3]
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()


