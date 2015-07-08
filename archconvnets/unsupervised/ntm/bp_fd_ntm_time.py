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
x3 = np.random.normal(size=(n_in,1))

do_dw3i = np.zeros((C,M,C,n2))
do_dw2i = np.zeros((C,M,n2,n1))
do_dw1i = np.zeros((C,M,n1,n_in))

do_content_dw3 = np.zeros_like(do_dw3i)
do_content_dw2 = np.zeros_like(do_dw2i)
do_content_dw1 = np.zeros_like(do_dw1i)

dmem_prev_dwwi = np.zeros((M, mem_length, C, M, n_in))

##############
def interpolate_simp_dx(dg3_dx, do_dx, do_content_dx, g3, o_prev, o_content, do_do_in):
	do_in_dx = np.einsum(do_dx + do_content_dx, range(4), g3, [0,3], range(4))
	do_in_dx += np.einsum(o_prev + o_content, [0,1], dg3_dx, [0,2,3], range(4))
	
	do_dx = mult_partials(do_do_in, do_in_dx, o_prev)
	return do_dx

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

##################
def sq_points(input):
	return input**2

def sq_points_dinput(input):
	n = input.shape[1]
	dinput = np.zeros((input.shape[0], n, input.shape[0], n))
	for i in range(input.shape[0]):
		dinput[i,range(n),i,range(n)] = 2*input[i]
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
	temp[range(M),:,:,range(M)] = add_out.T
	return temp

#######
def linear_2d_F(ww,x):
	return np.squeeze(np.dot(ww,x))

def linear_2d_F_dF_nsum(ww,x):
	n = ww.shape[1]
	temp = np.zeros((ww.shape[0], n, ww.shape[0], n, ww.shape[2]))
	for i in range(ww.shape[0]):
		temp[i,range(n),i,range(n)] += np.squeeze(x)
	return temp

def linear_2d_F_dx_nsum(ww):
	return ww

########
def forward_pass(w1,w2,w3,ww, o_prev, o_content, mem, mem_prev,x_cur):
	g1 = sq_F(w1,x_cur)
	g2 = sq_F(w2,g1)
	g3 = sq_F(w3,g2)
	o_in = interpolate_simp(o_prev, g3)
	o_in += interpolate_simp(o_content, g3)
	o_sq = sq_points(o_in)
	o = shift_w(shift_out, o_sq)
	read_mem = linear_F(o, mem_prev)
	
	gw = linear_2d_F(ww,x_cur)
	mem = mem_prev + add_mem(gw, add_out)
	
	return o,mem,read_mem,g1,g2,g3,o_in,o_sq,gw

##########
def compute_partials(w1,w2,w3,ww, o_prev, o_content, x_cur, x_prev, do_dw1, do_dw2, do_dw3, dmem_prev_dww,g1,g2,g3,o_in,o_sq,gw):
	## read gradients
	do_do_sq = shift_w_dw_interp_nsum(shift_out)
	do_sq_do_in = sq_points_dinput(o_in)
	do_do_in = mult_partials(do_do_sq, do_sq_do_in, o_sq)
	
	#w3
	dg3_dg2 = sq_dlayer_in_nsum(w3, g2)
	dg3_dw3 = sq_dF_nsum(w3, g2, g3)
	
	do_dw3 = interpolate_simp_dx(dg3_dw3, do_dw3, do_content_dw3, g3, o_prev, o_content, do_do_in)
	
	# w2
	dg2_dg1 = sq_dlayer_in_nsum(w2, g1)
	dg2_dw2 = sq_dF_nsum(w2, g1, g2)
	dg3_dw2 = mult_partials(dg3_dg2, dg2_dw2, np.squeeze(g2))
	do_dw2 = interpolate_simp_dx(dg3_dw2, do_dw2, do_content_dw2, g3, o_prev, o_content, do_do_in)
	
	# w1:
	dg1_dw1 = sq_dF_nsum(w1, x_cur, g1)
	dg3_dg1 = mult_partials(dg3_dg2, dg2_dg1, np.squeeze(g2))
	dg3_dw1 = mult_partials(dg3_dg1, dg1_dw1, np.squeeze(g1))
	do_dw1 = interpolate_simp_dx(dg3_dw1, do_dw1, do_content_dw1, g3, o_prev, o_content, do_do_in)
	
	## write gradients
	da_dgw = add_mem_dgw(add_out)
	dgw_dww = linear_2d_F_dF_nsum(ww,x_prev)
	
	da_dww = mult_partials(da_dgw, dgw_dww, gw)
	
	dmem_prev_dww += da_dww
	
	return do_dw1, do_dw2, do_dw3, dmem_prev_dww

def f(y):
	#w1[i_ind,j_ind] = y
	ww[i_ind,j_ind,k_ind] = y
	
	o_prev = copy.deepcopy(o_previ)
	mem_prev = copy.deepcopy(mem_previ)
	mem = np.zeros_like(mem_prev)
	
	# t1
	x_cur = copy.deepcopy(x)
	o_prev, mem_prev = forward_pass(w1,w2,w3,ww, o_prev, o_content, mem, mem_prev, x_cur)[:2]
	
	# t2
	x_cur = copy.deepcopy(x2)
	o_prev, mem_prev = forward_pass(w1,w2,w3,ww, o_prev, o_content, mem, mem_prev, x_cur)[:2]
	
	# t3
	x_cur = copy.deepcopy(x3)
	o_prev, mem_prev, read_mem = forward_pass(w1,w2,w3,ww, o_prev, o_content, mem, mem_prev, x_cur)[:3]
	
	return ((read_mem - t)**2).sum()


def g(y):
	#w1[i_ind,j_ind] = y
	ww[i_ind,j_ind,k_ind] = y
	
	x_prev = np.zeros_like(x)
	do_dw3 = copy.deepcopy(do_dw3i)
	do_dw2 = copy.deepcopy(do_dw2i)
	do_dw1 = copy.deepcopy(do_dw1i)
	o_prev = copy.deepcopy(o_previ)
	
	dmem_prev_dww = copy.deepcopy(dmem_prev_dwwi)
	mem_prev = copy.deepcopy(mem_previ)
	mem = np.zeros_like(mem_prev)
	
	# t1
	x_cur = copy.deepcopy(x)
	o,mem,read_mem,g1,g2,g3,o_in,o_sq,gw = forward_pass(w1,w2,w3,ww, o_prev, o_content, mem, mem_prev, x_cur)
	do_dw1, do_dw2, do_dw3, dmem_prev_dww = compute_partials(w1,w2,w3,ww, o_prev, o_content, \
			x_cur, x_prev, do_dw1, do_dw2, do_dw3, dmem_prev_dww,g1,g2,g3,o_in,o_sq,gw)
	o_prev = copy.deepcopy(o); mem_prev = copy.deepcopy(mem); x_prev = copy.deepcopy(x_cur)
	
	# t2
	x_cur = copy.deepcopy(x2)
	o,mem,read_mem,g1,g2,g3,o_in,o_sq,gw = forward_pass(w1,w2,w3,ww, o_prev, o_content, mem, mem_prev, x_cur)
	do_dw1, do_dw2, do_dw3, dmem_prev_dww = compute_partials(w1,w2,w3,ww, o_prev, o_content, \
			x_cur, x_prev, do_dw1, do_dw2, do_dw3, dmem_prev_dww,g1,g2,g3,o_in,o_sq,gw)
	o_prev = copy.deepcopy(o); mem_prev = copy.deepcopy(mem); x_prev = copy.deepcopy(x_cur)
	
	# t3
	x_cur = copy.deepcopy(x3)
	o,mem,read_mem,g1,g2,g3,o_in,o_sq,gw = forward_pass(w1,w2,w3,ww, o_prev, o_content, mem, mem_prev, x_cur)
	do_dw1, do_dw2, do_dw3, dmem_prev_dww = compute_partials(w1,w2,w3,ww, o_prev, o_content, \
			x_cur, x_prev, do_dw1, do_dw2, do_dw3, dmem_prev_dww,g1,g2,g3,o_in,o_sq,gw)
	
	dread_mem_do = linear_F_dF_nsum(mem_prev)
	dread_mem_dmem_prev = linear_F_dx_nsum(o)

	###
	derr_dread_mem = sq_points_dinput(read_mem - t)
	
	dread_mem_do = linear_F_dF_nsum(mem_prev)
	dread_mem_dmem_prev = linear_F_dx_nsum(o)
	
	derr_do = mult_partials(derr_dread_mem, dread_mem_do, read_mem)
	derr_dmem_prev = mult_partials(derr_dread_mem, dread_mem_dmem_prev, read_mem)
	
	dww = mult_partials_sum(derr_dmem_prev, dmem_prev_dww, mem_prev)
	
	dw1 = mult_partials_sum(derr_do, do_dw1, o)
	dw2 = mult_partials_sum(derr_do, do_dw2, o)
	dw3 = mult_partials_sum(derr_do, do_dw3, o)
	
	#return dw1[i_ind,j_ind]
	return dww[i_ind,j_ind,k_ind]
	
	
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e2


N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
for sample in range(N_SAMPLES):

	ref = ww
	#i_ind = np.random.randint(ref.shape[0])
	#j_ind = np.random.randint(ref.shape[1])
	#y = -1e0*ref[i_ind,j_ind]; gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
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


