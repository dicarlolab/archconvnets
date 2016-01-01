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

def free_all_buffers():
	for gpu_ind in range(N_GPUS):
		for buffer_ind in np.nonzero(n_vars_allocated[gpu_ind])[0]:
			_ntm_module2.free_buffer(buffer_ind, gpu_ind)
		n_vars_allocated[gpu_ind] = False

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

'''def add_points_dinput(args):
	assert len(args) == 2
	assert args[0].shape == args[1].shape
	out = np.zeros(np.concatenate((args[0].shape, args[0].shape)),dtype='single')
	for i in range(out.shape[0]):
		out[i,range(out.shape[1]),i,range(out.shape[1])] = 1
	return out'''

