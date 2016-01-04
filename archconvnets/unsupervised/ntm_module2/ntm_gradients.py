import numpy as np
import _ntm_module2
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *
from archconvnets.unsupervised.ntm2.gpu_flag import *

def linear_F_dx(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) == len(X[1]) == 2
	assert F[1][1] == X[1][0]
	
	F_dim0, F_dim1 = F[1]
	X_dim0, X_dim1 = X[1]
	
	if GPU:
		_ntm_module2.linear_F_dx(F[0], X[1], F[1], OUT_BUFFER[0], gpu_ind)
	else: 
		############ CPU
		F = return_buffer(F, gpu_ind)
		x = return_buffer(X, gpu_ind)
		n = x.shape[1]
		temp = np.zeros((F.shape[0], n, x.shape[0], n),dtype='single')
		temp[:,range(n),:,range(n)] = F
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = (F_dim0, X_dim1, X_dim0, X_dim1)
	return OUT_BUFFER

def linear_F_dF(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) == len(X[1]) == 2
	assert F[1][1] == X[1][0]
	
	F_dim0, F_dim1 = F[1]
	X_dim0, X_dim1 = X[1]
	
	if GPU:
		_ntm_module2.linear_F_dF(X[0], X[1], F[1], OUT_BUFFER[0], gpu_ind)
	else:
		############ CPU
		F = return_buffer(F, gpu_ind)
		x = return_buffer(X, gpu_ind)
		n = F.shape[0]
		temp = np.zeros((n, x.shape[1], n, F.shape[1]),dtype='single')
		temp[range(n),:,range(n)] = x.T
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (F_dim0, X_dim1, F_dim0, X_dim0)
	return OUT_BUFFER
	
def linear_F(args, OUT_BUFFER=None, increment=0, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	BUFFER1, BUFFER2 = args
	check_buffer(BUFFER1)
	check_buffer(BUFFER2)
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	check_buffer(OUT_BUFFER)
	assert len(BUFFER1[1]) == len(BUFFER2[1]) == 2
	assert BUFFER1[1][1] == BUFFER2[1][0]
	assert OUT_BUFFER[0] != BUFFER1[0]
	assert OUT_BUFFER[0] != BUFFER2[0]
	assert (OUT_BUFFER[1] is not None) or increment == 0
	
	if GPU:
		_ntm_module2.dot(BUFFER1[0], BUFFER1[1], BUFFER2[0], BUFFER2[1], OUT_BUFFER[0], increment, gpu_ind)
	else:
		######### CPU
		F = return_buffer(BUFFER1, gpu_ind)
		x = return_buffer(BUFFER2, gpu_ind)
		temp = np.asarray(np.dot(F,x),dtype='single') # [n1, 1]
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (BUFFER1[1][0], BUFFER2[1][1])
	return OUT_BUFFER

dot = linear_F

def sum_points(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	POINTS = args[0]
	check_buffer(POINTS)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module2.sum_points(POINTS[0], np.prod(POINTS[1]), OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		OUT_BUFFER = set_buffer(return_buffer(POINTS,gpu_ind).sum(), OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (1,)
	return OUT_BUFFER

def sum_points_dinput(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert len(args) == 1
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	POINTS = args[0]
	check_buffer(POINTS)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module2.sum_points_dinput(POINTS[0], np.prod(POINTS[1]), OUT_BUFFER[0], gpu_ind)
	else:
		######### CPU
		temp = np.ones(tuple(np.concatenate(((1,), args[0][1]))),dtype='single')
		OUT_BUFFER = set_buffer(temp, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = tuple(np.concatenate(((1,), POINTS[1])))
	return OUT_BUFFER
	
# a += b * scalar
def point_wise_add(args, OUT_BUFFER=None, scalar=1, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	A, B = args
	check_buffer(A)
	check_buffer(B)
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = copy.deepcopy(A[1])
	else:
		OUT_BUFFER = copy.deepcopy(A)
	
	if GPU:
		_ntm_module2.point_wise_add(A[0], B[0], np.single(scalar), OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		A_local = return_buffer(A,gpu_ind)
		B_local = return_buffer(B,gpu_ind)
		OUT_BUFFER = set_buffer(A_local + B_local*scalar, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = copy.deepcopy(B[1])
	return OUT_BUFFER

# unlike point_wise_add, this defaults to storing the output in a new buffer instead of overwriting the first argument
def add_points(args, OUT_BUFFER=None, scalar=1, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	A, B = args
	check_buffer(A)
	check_buffer(B)
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = copy.deepcopy(A[1])
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		_ntm_module2.point_wise_add(A[0], B[0], np.single(scalar), OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		A_local = return_buffer(A,gpu_ind)
		B_local = return_buffer(B,gpu_ind)
		OUT_BUFFER = set_buffer(A_local + B_local*scalar, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = copy.deepcopy(B[1])
	return OUT_BUFFER

def add_points_dinput(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	A, B = args
	check_buffer(A)
	check_buffer(B)
	check_buffer(LAYER_OUT)
	assert A[1] == B[1]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	if GPU:
		_ntm_module2.add_points_dinput(A[1], OUT_BUFFER[0], gpu_ind)
	else:
		######### CPU
		out = np.zeros(np.concatenate((args[0][1], args[0][1])), dtype='single')
		for i in range(out.shape[0]):
			out[i,range(out.shape[1]),i,range(out.shape[1])] = 1
		OUT_BUFFER = set_buffer(out, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((A[1], A[1])))
	return OUT_BUFFER
	
def cosine_sim(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	KEYS, MEM = args
	check_buffer(KEYS)
	check_buffer(MEM)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(KEYS[1]) == len(MEM[1]) == 2
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
		numer = np.dot(keys, mem.T)
		denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
		OUT_BUFFER = set_buffer(numer/denom, OUT_BUFFER, gpu_ind)
	
	OUT_BUFFER[1] = (n_controllers, M)
	return OUT_BUFFER

def cosine_sim_dmem(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	KEYS, MEM = args
	check_buffer(KEYS)
	check_buffer(MEM)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(KEYS[1]) == len(MEM[1]) == 2
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim_dmem(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	else:
		########## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
		comb = np.zeros((n_controllers, mem.shape[0], mem.shape[0], mem.shape[1]),dtype='single')

		keys_sq_sum = np.sqrt(np.sum(keys**2, 1))
		mem_sq_sum = np.sqrt(np.sum(mem**2, 1))

		denom = np.einsum(keys_sq_sum, [0], mem_sq_sum, [1], [0,1])
		numer = np.dot(keys, mem.T)

		numer = numer / denom**2
		denom = 1 / denom # = denom/denom**2

		mem = mem / mem_sq_sum[:,np.newaxis]
		temp = np.einsum(mem, [0,2], numer*keys_sq_sum[:,np.newaxis], [1,0], [1,0,2])
		keys_denom = keys[:,np.newaxis] * denom[:,:,np.newaxis]
		
		comb[:,range(mem.shape[0]),range(mem.shape[0])] = keys_denom - temp
		OUT_BUFFER = set_buffer(comb, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (n_controllers, M, M, mem_length)
	return OUT_BUFFER

def cosine_sim_dkeys(args, LAYER_OUT, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	KEYS, MEM = args
	check_buffer(KEYS)
	check_buffer(MEM)
	check_buffer(LAYER_OUT)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(KEYS[1]) == len(MEM[1]) == 2
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers, mem_length = KEYS[1]
	M = MEM[1][0]
	
	if GPU:
		_ntm_module2.cosine_sim_dkeys(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	else:
		######## CPU
		keys = return_buffer(KEYS, gpu_ind)
		mem = return_buffer(MEM, gpu_ind)
		comb = np.zeros((n_controllers, mem.shape[0], n_controllers, keys.shape[1]),dtype='single')
		
		keys_sq_sum = np.sqrt(np.sum(keys**2, 1))
		mem_sq_sum = np.sqrt(np.sum(mem**2, 1))
		
		denom = np.einsum(keys_sq_sum, [0], mem_sq_sum, [1], [0,1])
		numer = np.dot(keys, mem.T)
		
		numer = numer / denom**2
		denom = 1 / denom # = denom/denom**2
		
		keys = keys / keys_sq_sum[:,np.newaxis]
		temp = np.einsum(keys, [1,2], numer*mem_sq_sum[np.newaxis], [1,0], [1,0,2])
		mem_denom = mem[np.newaxis] * denom[:,:,np.newaxis]
		
		comb[range(n_controllers),:,range(n_controllers)] = mem_denom - temp
		OUT_BUFFER = set_buffer(comb, OUT_BUFFER, gpu_ind)
		
	OUT_BUFFER[1] = (n_controllers, M, n_controllers, mem_length)
	return OUT_BUFFER

