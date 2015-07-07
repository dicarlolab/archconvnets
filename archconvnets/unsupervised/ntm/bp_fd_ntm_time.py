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

SCALE = .4

t = np.random.normal(size=(C,mem_length))

mem_previ = np.random.normal(size=(M, mem_length))

o_previ = np.random.normal(size=(C,M))
o_content = np.random.normal(size=(C,M))

w3 = np.random.normal(size=(C,n2)) * SCALE
w2 = np.random.normal(size=(n2,n1)) * SCALE
w1 = np.random.normal(size=(n1,n_in)) * SCALE

ww = np.random.normal(size=(C, M, n_in)) * SCALE

add_out = np.random.normal(size=(C, mem_length)) * SCALE

shift_out = np.random.normal(size=(C, n_shifts))

x = np.random.normal(size=(n_in,1))
x2 = np.random.normal(size=(n_in,1))

do_dw3i = np.zeros((C,M,C,n2))
do_dw2i = np.zeros((C,M,n2,n1))
do_dw1i = np.zeros((C,M,n1,n_in))

do_content_dw3 = np.zeros_like(do_dw3i)
do_content_dw2 = np.zeros_like(do_dw2i)
do_content_dw1 = np.zeros_like(do_dw1i)

dmem_prev_dwwi = np.zeros((M, mem_length, C, M, n_in))

##########
def update_partials(g1,g2,g3,w1,w2,w3,x,o_prev,o_content,do_do_sq,do_do_in, do_dw1,do_dw2,do_dw3):
	# w3:
	dg3_dg2 = sq_dlayer_in_nsum(w3, g2)
	dg3_dw3 = sq_dF_nsum(w3, g2, g3)
	
	do_dw3 = interpolate_simp_dx(dg3_dw3, do_dw3, do_content_dw3, g3, o_prev, o_content, do_do_in)
	
	# w2:
	dg2_dg1 = sq_dlayer_in_nsum(w2, g1)
	dg2_dw2 = sq_dF_nsum(w2, g1, g2)
	dg3_dw2 = np.einsum(dg3_dg2,[0,1], dg2_dw2, [1,2,3], [0,2,3])
	do_dw2 = interpolate_simp_dx(dg3_dw2, do_dw2, do_content_dw2, g3, o_prev, o_content, do_do_in)
	
	# w1:
	dg1_dw1 = sq_dF_nsum(w1, x, g1)
	dg3_dg1 = np.einsum(dg3_dg2, [0,1], dg2_dg1, [1,2], [0,1,2])
	dg3_dw1 = np.einsum(dg3_dg1, [0,1,2], dg1_dw1, [2,3,4], [0,3,4])
	do_dw1 = interpolate_simp_dx(dg3_dw1, do_dw1, do_content_dw1, g3, o_prev, o_content, do_do_in)
	
	return do_dw1, do_dw2, do_dw3

##############
def interpolate_simp_dx(dg3_dx, do_dx, do_content_dx, g3, o_prev, o_content, do_do_in):
	do_in_dx = np.einsum(do_dx + do_content_dx, range(4), g3, [0,3], range(4))
	do_in_dx += np.einsum(o_prev + o_content, [0,1], dg3_dx, [0,2,3], range(4))
	
	do_dx = np.einsum(do_do_in, range(4), do_in_dx, [2,3,4,5], [0,1,4,5])
	return do_dx

############
def read_from_mem_dmem_nsum(o):
	temp = np.zeros((o_previ.shape[0], mem_previ.shape[1], mem_previ.shape[0], mem_previ.shape[1]))
	for i in range(o_previ.shape[0]):
		for k in range(o_previ.shape[1]):
			for l in range(mem_previ.shape[1]):
				temp[i,l, k,l] += o[i,k]
	return temp

def read_from_mem_dw_nsum(mem):
	temp = np.zeros((o_previ.shape[0], mem.shape[1], o_previ.shape[0], o_previ.shape[1]))
	for i in range(o_previ.shape[0]):
		for k in range(o_previ.shape[1]):
			for l in range(mem.shape[1]):
				temp[i,l, i,k] += mem[k,l]
	return temp

##################
def sq_points(input):
	return input**2

def sq_points_dinput_comb(input, above_layer):
	dinput = np.zeros((input.shape[0], input.shape[1], input.shape[0], input.shape[1]))
	for i in range(input.shape[0]):
		for j in range(input.shape[1]):
			dinput[i,j,i,j] = 2*input[i,j]
	
	return np.einsum(above_layer, [0,1,2,3], dinput, [2,3,4,5], [0,1,4,5])

def sq_points_dinput(input):
	dinput = np.zeros((input.shape[0], input.shape[1], input.shape[0], input.shape[1]))
	for i in range(input.shape[0]):
		for j in range(input.shape[1]):
			dinput[i,j,i,j] = 2*input[i,j]
	return dinput

################# interpolate simplified
def interpolate_simp(w_prev, interp_gate_out):
	return w_prev * interp_gate_out
	
############### shift w
def shift_w(shift_out, w_interp):	
	# shift_out: [n_controllers, n_shifts], w_interp: [n_controllers, n_mem_slots]
	
	w_tilde = np.zeros_like(w_interp)
	n_mem_slots = w_interp.shape[1]
	
	for loc in range(n_mem_slots):
		w_tilde[:,loc] = shift_out[:,0]*w_interp[:,loc-1] + shift_out[:,1]*w_interp[:,loc] + \
				shift_out[:,2]*w_interp[:,(loc+1)%n_mem_slots]
	return w_tilde # [n_controllers, n_mem_slots]
	
################
def shift_w_dw_interp_nsum(shift_out):
	# shift_out: [n_controllers, n_shifts]
	
	temp = np.zeros((C, M, C, M))
	
	for c in range(C):
		for loc in range(M):
			temp[c,loc,c,loc-1] = shift_out[c,0]
			temp[c,loc,c,loc] = shift_out[c,1]
			temp[c,loc,c,(loc+1)%M] = shift_out[c,2]
			
	return temp # [n_controllers, M, n_controllers, M]

#####
def add_mem(gw, add_out):
	return np.dot(gw.T, add_out)

def add_mem_dgw(add_out):
	temp = np.zeros((M, mem_length, C, M))
	
	for i in range(C):
		for k in range(M):
			for j in range(mem_length):
				temp[k,j,i,k] += add_out[i,j]# * gw[i,k]
	return temp

#######
def linear_2d_F(ww,x):
	return np.squeeze(np.dot(ww,x))

def linear_2d_F_dF_nsum(ww,x):
	temp = np.zeros((ww.shape[0], ww.shape[1], ww.shape[0], ww.shape[1], ww.shape[2]))
	for i in range(ww.shape[0]):
		for j in range(ww.shape[1]):
			for k in range(ww.shape[2]):
				temp[i,j,i,j,k] += x[k]
	return temp

def linear_2d_F_dx_nsum(ww,x):
	temp = np.zeros((ww.shape[0], ww.shape[1], x.shape[0]))
	for i in range(ww.shape[0]):
		for j in range(ww.shape[1]):
			for k in range(ww.shape[2]):
				temp[i,j,k] += ww[i,j,k]
	return temp

def f(y):
	#w3[i_ind,j_ind] = y
	ww[i_ind,j_ind,k_ind] = y
	
	o_prev = copy.deepcopy(o_previ)
	mem_prev = copy.deepcopy(mem_previ)
	
	###
	g1 = sq_F(w1,x)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o_sq = sq_points(o_in)
	o = shift_w(shift_out, o_sq)
	read_mem = read_from_mem(o, mem_prev)
	
	gw = linear_2d_F(ww,x)
	mem = mem_prev + add_mem(gw, add_out)
	
	o_prev = copy.deepcopy(o)
	mem_prev = copy.deepcopy(mem)
	
	###
	g1 = sq_F(w1,x2)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o_sq = sq_points(o_in)
	o = shift_w(shift_out, o_sq)
	read_mem = read_from_mem(o, mem_prev)
	
	gw = linear_2d_F(ww,x2)
	mem = mem_prev + add_mem(gw, add_out)
	
	o_prev = copy.deepcopy(o)
	mem_prev = copy.deepcopy(mem)
	
	return ((read_mem - t)**2).sum()


def g(y):
	#w3[i_ind,j_ind] = y
	ww[i_ind,j_ind,k_ind] = y
	
	do_dw3 = copy.deepcopy(do_dw3i)
	do_dw2 = copy.deepcopy(do_dw2i)
	do_dw1 = copy.deepcopy(do_dw1i)
	o_prev = copy.deepcopy(o_previ)
	
	dmem_prev_dww = copy.deepcopy(dmem_prev_dwwi)
	mem_prev = copy.deepcopy(mem_previ)
	
	
	### forward
	g1 = sq_F(w1,x)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o_sq = sq_points(o_in)
	o = shift_w(shift_out, o_sq)
	read_mem = read_from_mem(o, mem_prev)
	
	gw = linear_2d_F(ww,x)
	a = add_mem(gw, add_out)
	mem = mem_prev + a
	
	# read gradients
	dread_mem_do = read_from_mem_dw_nsum(mem_prev)
	
	do_do_sq = shift_w_dw_interp_nsum(shift_out)
	do_do_in = sq_points_dinput_comb(o_in, do_do_sq)
	
	do_dw1, do_dw2, do_dw3 = update_partials(g1,g2,g3,w1,w2,w3,x,o_prev,o_content,do_do_sq,do_do_in, do_dw1,do_dw2,do_dw3)
	
	##
	o_prev = copy.deepcopy(o)
	mem_prev = copy.deepcopy(mem)
	x_prev = copy.deepcopy(x)
	
	### forward
	g1 = sq_F(w1,x2)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o_sq = sq_points(o_in)
	o = shift_w(shift_out, o_sq)
	read_mem = read_from_mem(o, mem_prev)
	
	gw = linear_2d_F(ww,x2)
	a = add_mem(gw, add_out)
	mem = mem_prev + a
	
	# read gradients
	dread_mem_do = read_from_mem_dw_nsum(mem_prev)
	
	do_do_sq = shift_w_dw_interp_nsum(shift_out)
	do_do_in = sq_points_dinput_comb(o_in, do_do_sq)
	
	do_dw1, do_dw2, do_dw3 = update_partials(g1,g2,g3,w1,w2,w3,x2,o_prev,o_content,do_do_sq,do_do_in, do_dw1,do_dw2,do_dw3)
	
	# write gradients
	dread_mem_dmem_prev = read_from_mem_dmem_nsum(o)
	
	da_dgw = add_mem_dgw(add_out)
	dgw_dww = linear_2d_F_dF_nsum(ww,x_prev)
	
	da_dww = np.einsum(da_dgw, range(4), dgw_dww, [2,3,4,5,6], [0,1, 4,5,6])
	
	dmem_prev_dww += da_dww
	
	######
	o_prev = copy.deepcopy(o)
	mem_prev = copy.deepcopy(mem)
	x_prev = copy.deepcopy(x)
	
	
	###
	derr_dread_mem = 2*(read_mem - t)
	
	derr_do = np.einsum(derr_dread_mem, [0,1], dread_mem_do, range(4), range(4))
	derr_dmem_prev = np.einsum(derr_dread_mem, [0,1], dread_mem_dmem_prev, range(4), range(4))
	
	dww = np.einsum(derr_dmem_prev, range(4), dmem_prev_dww, [2,3,4,5,6], [4,5,6])
	
	dw1 = np.einsum(derr_do, range(4), do_dw1,  [2,3,4,5], [4,5])
	dw2 = np.einsum(derr_do, range(4), do_dw2,  [2,3,4,5], [4,5])
	dw3 = np.einsum(derr_do, range(4), do_dw3,  [2,3,4,5], [4,5])
	
	#return dw3[i_ind,j_ind]
	return dww[i_ind,j_ind,k_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = ww
	'''i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)'''
	
	i_ind = np.random.randint(ref.shape[0])
	j_ind = np.random.randint(ref.shape[1])
	k_ind = np.random.randint(ref.shape[2])
	y = -1e0*ref[i_ind,j_ind,k_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
		
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std()


