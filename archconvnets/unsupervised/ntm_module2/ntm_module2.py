import _ntm_module2
import numpy as np
import copy

N_GPUS = 4
N_BUFFERS = 1024
n_vars_allocated = np.zeros((N_GPUS, N_BUFFERS), dtype='bool') # variable slots allocated per gpu

def sync(gpu_ind=0):
	assert isinstance(gpu_ind,int)
	return _ntm_module2.sync(gpu_ind)

def return_buffer_sz(buffer_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(buffer_ind,int)
	return _ntm_module2.return_buffer_sz(buffer_ind, gpu_ind)

def check_buffer(BUFFER, gpu_ind=0):
	assert len(BUFFER) == 2
	assert isinstance(BUFFER[0], int)
	assert BUFFER[0] >= 0
	assert n_vars_allocated[gpu_ind, BUFFER[0]]
	assert isinstance(BUFFER[1], tuple) or BUFFER[1] == None
	if BUFFER[1] is not None:
		assert return_buffer_sz(BUFFER[0], gpu_ind) == np.prod(BUFFER[1])
	else:
		assert return_buffer_sz(BUFFER[0], gpu_ind) == 0

def free_buffer(BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(BUFFER)
	n_vars_allocated[gpu_ind, BUFFER[0]] = False
	_ntm_module2.free_buffer(BUFFER[0], gpu_ind)
	BUFFER = [-1, None]

def return_buffer(BUFFER, warn=1, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(BUFFER)

	return _ntm_module2.return_buffer(BUFFER[0], warn, gpu_ind).reshape(BUFFER[1])

def return_n_allocated(gpu_ind=0):
	return n_vars_allocated[gpu_ind].sum()

def set_buffer(DATA, DATA_G, gpu_ind=0, warn=True):
	assert DATA.dtype == np.dtype('float32')
	if not DATA.flags.contiguous and warn:
		print 'warning: input to init_buffer not C-contiguous'
		DATA = np.ascontiguousarray(DATA)
	_ntm_module2.set_buffer(DATA, DATA_G[0], gpu_ind)
	DATA_G[1] = DATA.shape

def init_buffer(DATA=None, gpu_ind=0, warn=True):
	assert isinstance(gpu_ind,int)
	z = np.nonzero(1-n_vars_allocated[gpu_ind])[0]
	assert len(z) != 0, 'no memory slots left'
	buffer_ind = z.min()
	if DATA is not None:
		DATA_G = [buffer_ind, DATA.shape]
		set_buffer(DATA, DATA_G, gpu_ind=gpu_ind, warn=warn)
	else:
		DATA_G = [buffer_ind, None]
	n_vars_allocated[gpu_ind,buffer_ind] = True
	return DATA_G

# copy B to A
def copy_buffer(B, A=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(B)
	if A is None:
		A = init_buffer()
	check_buffer(A)
	_ntm_module2.copy_buffer(B[0], A[0], gpu_ind)
	A[1] = copy.deepcopy(B[1])
	return A

def copy_list(LIST_B, LIST_A=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	if LIST_A is None:
		LIST_A = [None]*len(LIST_B)
	
	assert len(LIST_A) == len(LIST_B)
	
	for i in range(len(LIST_B)):
		LIST_A[i] = copy_buffer(LIST_B[i], LIST_A[i], gpu_ind)
	return LIST_A

def free_all_buffers():
	for gpu_ind in range(N_GPUS):
		for buffer_ind in np.nonzero(n_vars_allocated[gpu_ind])[0]:
			_ntm_module2.free_buffer(buffer_ind, gpu_ind)
		n_vars_allocated[gpu_ind] = False

def free_list(LIST):
	for layer_ind in range(len(LIST)):
		if LIST[layer_ind] is not None:
			# list of list, ex. partials
			if isinstance(LIST[layer_ind][0], list):
				for arg in range(len(LIST[layer_ind])):
					free_buffer(LIST[layer_ind][arg])
			# outputs
			else:
				free_buffer(LIST[layer_ind])

#######################################

def linear_F_dx(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) == len(X[1]) == 2
	assert F[1][1] == X[1][0]
	
	F_dim0, F_dim1 = F[1]
	X_dim0, X_dim1 = X[1]
	
	_ntm_module2.linear_F_dx(F[0], X[1], F[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (F_dim0, X_dim1, X_dim0, X_dim1)
	return OUT_BUFFER

def linear_F_dF(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, X = args
	check_buffer(F)
	check_buffer(X)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) == len(X[1]) == 2
	assert F[1][1] == X[1][0]
	
	F_dim0, F_dim1 = F[1]
	X_dim0, X_dim1 = X[1]
	
	_ntm_module2.linear_F_dF(X[0], X[1], F[1], OUT_BUFFER[0], gpu_ind)
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
	
	_ntm_module2.dot(BUFFER1[0], BUFFER1[1], BUFFER2[0], BUFFER2[1], OUT_BUFFER[0], increment, gpu_ind)
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
	
	_ntm_module2.sum_points(POINTS[0], np.prod(POINTS[1]), OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (1,)
	return OUT_BUFFER

def sum_points_dinput(args, OUT_BUFFER=None, gpu_ind=0):
	assert len(args) == 1
	assert isinstance(gpu_ind,int)
	assert len(args) == 1
	POINTS = args[0]
	check_buffer(POINTS)
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	
	_ntm_module2.sum_points_dinput(POINTS[0], np.prod(POINTS[1]), OUT_BUFFER[0], gpu_ind)
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
	
	_ntm_module2.point_wise_add(A[0], B[0], np.single(scalar), OUT_BUFFER[0], gpu_ind)
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
	
	_ntm_module2.point_wise_add(A[0], B[0], np.single(scalar), OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = copy.deepcopy(B[1])
	return OUT_BUFFER

def add_points_dinput(args, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	A, B = args
	check_buffer(A)
	check_buffer(B)
	assert A[1] == B[1]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	check_buffer(OUT_BUFFER)
	_ntm_module2.add_points_dinput(A[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = tuple(np.concatenate((A[1], A[1])))
	return OUT_BUFFER

'''
def cosine_sim_dmem(args):
	assert len(args) == 2
	keys, mem = args
	n_controllers = keys.shape[0]
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
	return comb

def cosine_sim_dkeys(args):
	assert len(args) == 2
	keys, mem = args
	n_controllers = keys.shape[0]
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
	return comb

def cosine_sim(args):
	assert len(args) == 2
	keys, mem = args
	# keys [n_controllers, m_length], mem: [n_mem_slots, m_length]
	numer = np.dot(keys, mem.T)
	denom = np.einsum(np.sqrt(np.sum(keys**2,1)), [0], np.sqrt(np.sum(mem**2,1)), [1], [0,1])
	return numer / denom # [n_controllers, n_mem_slots]'''
	
	
def cosine_sim_dmem(args, OUT_BUFFER=None, gpu_ind=0):
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

	n_controllers = KEYS[1][0]
	mem_length = KEYS[1][1]
	M = MEM[1][0]
	
	_ntm_module2.cosine_sim_dmem(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (n_controllers, M, M, mem_length)
	return OUT_BUFFER

def cosine_sim_dkeys(args, OUT_BUFFER=None, gpu_ind=0):
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

	n_controllers = KEYS[1][0]
	mem_length = KEYS[1][1]
	M = MEM[1][0]
	
	_ntm_module2.cosine_sim_dkeys(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (n_controllers, M, n_controllers, mem_length)
	return OUT_BUFFER
