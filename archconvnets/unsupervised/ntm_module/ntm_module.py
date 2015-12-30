import _ntm_module
import numpy as np
import copy

n_vars_allocated = np.zeros(4, dtype='int') # variable slots allocated per gpu

def check_buffer(BUFFER):
	assert len(BUFFER) == 2
	assert isinstance(BUFFER[0], int)
	assert BUFFER[0] >= 0
	assert isinstance(BUFFER[1], tuple) or BUFFER[1] == None

def mult_partials(DA_DB, DB_DC, B, OUT_BUFFER, increment=0, squeeze=0, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(increment,int)
	check_buffer(DA_DB)
	check_buffer(DB_DC)
	check_buffer(OUT_BUFFER)
	check_buffer(B)
	assert (OUT_BUFFER[1] is not None) or increment == 0
	assert OUT_BUFFER[0] != DA_DB[0] and OUT_BUFFER[0] != DB_DC[0]
	B_shape = np.asarray(B[1])
	
	da_db_shape = np.asarray(DA_DB[1])
	db_dc_shape = np.asarray(DB_DC[1])
	
	if squeeze == 1:
		da_db_shape = da_db_shape[da_db_shape != 1]
		db_dc_shape = db_dc_shape[db_dc_shape != 1]
		B_shape = B_shape[B_shape != 1]
	
	b_ndim = len(B_shape)
	
	a_ndim = len(da_db_shape) - b_ndim
	c_ndim = len(db_dc_shape) - b_ndim
	
	assert c_ndim > 0
	assert (db_dc_shape[:b_ndim] == B_shape).sum() == b_ndim
	#assert a_ndim >
	#if a_ndim <= 0:
	#	print da_db_shape, db_dc_shape, B_shape
	#	print (np.prod(da_db_shape[:a_ndim]), np.prod(da_db_shape[a_ndim:])), (np.prod(db_dc_shape[:b_ndim]), np.prod(db_dc_shape[b_ndim:]))
	#	print 
	
	da_db_shape = (np.prod(da_db_shape[:a_ndim]), np.prod(da_db_shape[a_ndim:]))
	db_dc_shape = (np.prod(db_dc_shape[:b_ndim]), np.prod(db_dc_shape[b_ndim:]))
	
	_ntm_module.dot(DA_DB[0], da_db_shape, DB_DC[0], db_dc_shape, OUT_BUFFER[0], increment, gpu_ind)
	OUT_BUFFER[1] = (da_db_shape[0], db_dc_shape[1])

# multiply list of partials
def mult_partials_chain(L_DA_DB, B, L_OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert len(L_DA_DB) == len(L_OUT_BUFFER)
	DA_DX = L_DA_DB[0]
	check_buffer(DA_DX)
	for x in range(1, len(L_DA_DB)):
		check_buffer(L_DA_DB[x])
		check_buffer(L_OUT_BUFFER[x])
		mult_partials(DA_DX, L_DA_DB[x], B[x-1], L_OUT_BUFFER[x])
		DA_DX = L_OUT_BUFFER[x]

# mult_partials for all layers in DB_DC (a list of indices)
def mult_partials__layers(DA_DB, L_DB_DC, B, OUT_BUFFER, increment=0, squeeze=0, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(increment,int)
	assert (OUT_BUFFER[1] is not None) or increment == 0
	check_buffer(DA_DB)
	
	assert len(L_DB_DC) == len(OUT_BUFFER)
	
	for l in range(len(L_DB_DC)):
		check_buffer(L_DB_DC[l])
		mult_partials(DA_DB, L_DB_DC[l], B, OUT_BUFFER[l], increment=increment, squeeze=squeeze, gpu_ind=gpu_ind)

def sq_points_dinput(INPUT, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(INPUT)
	check_buffer(OUT_BUFFER)
	assert len(INPUT[1]) == 2
	
	dim0, dim1 = INPUT[1]
	
	_ntm_module.sq_points_dinput(INPUT[0], INPUT[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (dim0, dim1, dim0, dim1)

# L_A[i] *= b * scalar
# L_A[i]: 4 dim
# b: 2 dim
def pointwise_mult_partials__layers(L_A, B, scalar=1, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(B)
	
	for i in range(len(L_A)):
		check_buffer(L_A[i])
		point_wise_mult_bcast2(L_A[i], B, scalar=scalar, gpu_ind=gpu_ind)
	
	
# a *= b * scalar
# a: 4 dim
# b: 2 dim
def point_wise_mult_bcast2(A, B, scalar=1, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(A)
	check_buffer(B)
	
	assert len(A[1]) > 2
	assert len(B[1]) == 2
	assert A[1][0] == B[1][0]
	assert A[1][1] == B[1][1]
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = copy.deepcopy(A[1])
	else:
		OUT_BUFFER = copy.deepcopy(A)
	
	_ntm_module.point_wise_mult_bcast2(A[0], A[1], B[0], np.single(scalar), OUT_BUFFER[0], gpu_ind)

# a = a * scalar1 + scalar2
def point_wise_add_scalar(A, scalar1=1, scalar2=1, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(A)
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = copy.deepcopy(A[1])
	else:
		OUT_BUFFER = copy.deepcopy(A)
	
	_ntm_module.point_wise_add_scalar(A[0], np.single(scalar1), np.single(scalar2), OUT_BUFFER[0], gpu_ind)


# a += b * scalar
def point_wise_add(A, B, scalar=1, OUT_BUFFER=None, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(A)
	check_buffer(B)
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = copy.deepcopy(A[1])
	else:
		OUT_BUFFER = copy.deepcopy(A)
	
	_ntm_module.point_wise_add(A[0], B[0], np.single(scalar), OUT_BUFFER[0], gpu_ind)
	

def shift_w_dw_interp(SHIFT_OUT, W_INTERP, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(SHIFT_OUT)
	check_buffer(W_INTERP)
	check_buffer(OUT_BUFFER)
	assert len(W_INTERP[1]) == 2
	C, M = W_INTERP[1]
	
	_ntm_module.shift_w_dw_interp(SHIFT_OUT[0], W_INTERP[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (C,M,C,M)
	

def shift_w_dshift_out(W_INTERP, OUT_BUFFER, gpu_ind=0):
	N_SHIFTS = 3
	assert isinstance(gpu_ind,int)
	assert isinstance(OUT_BUFFER[0],int)
	assert isinstance(W_INTERP[0],int)
	assert isinstance(W_INTERP[1],tuple)
	assert len(W_INTERP) == len(OUT_BUFFER) == 2
	assert len(W_INTERP[1]) == 2
	C, M = W_INTERP[1]
	
	_ntm_module.shift_w_dshift_out(W_INTERP[0], W_INTERP[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (C,M,C,N_SHIFTS)


def interpolate_dinterp_gate_out(O_CONTENT, O_PREV, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(O_PREV)
	check_buffer(O_CONTENT)
	check_buffer(OUT_BUFFER)	
	assert len(O_PREV[1]) == len(O_CONTENT[1]) == 2
	
	dim0, dim1 = O_CONTENT[1]
	
	_ntm_module.interpolate_dinterp_gate_out(O_CONTENT[0], O_CONTENT[1], O_PREV[0], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (dim0,dim1,dim0,1)

def interpolate_do_content(INTERP_GATE_OUT, O_PREV, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(INTERP_GATE_OUT)
	check_buffer(O_PREV)
	check_buffer(OUT_BUFFER)	
	assert len(O_PREV[1]) == 2
	
	dim0, dim1 = O_PREV[1]
	
	_ntm_module.interpolate_do_content(INTERP_GATE_OUT[0], O_PREV[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (dim0,dim1,dim0,dim1)


def interpolate_do_prev(INTERP_GATE_OUT, O_PREV, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(INTERP_GATE_OUT)
	check_buffer(O_PREV)
	check_buffer(OUT_BUFFER)	
	assert len(O_PREV[1]) == 2
	
	dim0, dim1 = O_PREV[1]
	
	_ntm_module.interpolate_do_prev(INTERP_GATE_OUT[0], O_PREV[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (dim0,dim1,dim0,dim1)

def interpolate_softmax_do_prev(OUT, INTERP_GATE_OUT, O_PREV, OUT_BUFFER):
	DOUT_DLIN = init_buffer()
	DLIN_DO_PREV = init_buffer()
	
	softmax_dlayer_in(OUT, DOUT_DLIN)
	interpolate_do_prev(INTERP_GATE_OUT, O_PREV, DLIN_DO_PREV)
	mult_partials(DOUT_DLIN, DLIN_DO_PREV, OUT, OUT_BUFFER)
	
	free_buffer(DOUT_DLIN[0])
	free_buffer(DLIN_DO_PREV[0])
	
def interpolate_softmax_dinterp_gate_out(OUT, O_CONTENT, O_PREV, OUT_BUFFER):
	DOUT_DLIN = init_buffer()
	DLIN_DINTERP_GATE_OUT = init_buffer()
	
	softmax_dlayer_in(OUT, DOUT_DLIN)
	interpolate_dinterp_gate_out(O_CONTENT, O_PREV, DLIN_DINTERP_GATE_OUT)
	mult_partials(DOUT_DLIN, DLIN_DINTERP_GATE_OUT, OUT, OUT_BUFFER)
	
	free_buffer(DOUT_DLIN[0])
	free_buffer(DLIN_DINTERP_GATE_OUT[0])
	
def interpolate_softmax_do_content(OUT, INTERP_GATE_OUT, O_PREV, OUT_BUFFER):
	DOUT_DLIN = init_buffer()
	DLIN_DO_CONTENT = init_buffer()
	
	softmax_dlayer_in(OUT, DOUT_DLIN)
	interpolate_do_content(INTERP_GATE_OUT, O_PREV, DLIN_DO_CONTENT)
	mult_partials(DOUT_DLIN, DLIN_DO_CONTENT, OUT, OUT_BUFFER)
	
	free_buffer(DOUT_DLIN[0])
	free_buffer(DLIN_DO_CONTENT[0])

def linear_F_dx(F, X, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(F)
	check_buffer(X)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) == len(X[1]) == 2
	assert F[1][1] == X[1][0]
	
	F_dim0, F_dim1 = F[1]
	X_dim0, X_dim1 = X[1]
	
	_ntm_module.linear_F_dx(F[0], X[1], F[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (F_dim0, X_dim1, X_dim0, X_dim1)

def linear_F_dF(F, X, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(X)
	check_buffer(F)
	check_buffer(OUT_BUFFER)
	assert len(F[1]) == len(X[1]) == 2
	assert F[1][1] == X[1][0]
	
	F_dim0, F_dim1 = F[1]
	X_dim0, X_dim1 = X[1]
	
	_ntm_module.linear_F_dF(X[0], X[1], F[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (F_dim0, X_dim1, F_dim0, X_dim0)

def linear_2d_F_dF(F, X, OUT_BUFFER, gpu_ind=0):
	# (16, 8, 16, 8, 9) (16, 8, 9) (9, 1)
	assert isinstance(gpu_ind,int)
	check_buffer(X)
	check_buffer(F)
	check_buffer(OUT_BUFFER)
	assert len(X[1]) == 2
	assert len(F[1]) == 3
	
	Fr = (F[1][0]*F[1][1], F[1][2])
	
	assert Fr[1] == X[1][0]
	
	F_dim0 = F[1][0]
	F_dim1 = F[1][1]
	X_dim0, X_dim1 = X[1]
	
	_ntm_module.linear_F_dF(X[0], X[1], Fr, OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (F_dim0, F_dim1, F_dim0, F_dim1, X_dim0)

def linear_2d_F_dx(F, X, OUT_BUFFER, gpu_ind=0):
	# (16, 8, 16, 8, 9) (16, 8, 9) (9, 1)
	assert isinstance(gpu_ind,int)
	check_buffer(X)
	check_buffer(F)
	check_buffer(OUT_BUFFER)
	assert len(X[1]) == 2
	assert len(F[1]) == 3
	
	Fr = (F[1][0]*F[1][1], F[1][2])
	
	assert Fr[1] == X[1][0]
	
	F_dim0 = F[1][0]
	F_dim1 = F[1][1]
	X_dim0, X_dim1 = X[1]
	
	_ntm_module.linear_F_dx(F[0], X[1], Fr, OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (F_dim0, F_dim1, X_dim0, X_dim1)
	
def relu_dlayer_in(LAYER_IN, OUT_BUFFER, thresh=0, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(LAYER_IN)
	check_buffer(OUT_BUFFER)
	assert isinstance(thresh,int)
	assert len(LAYER_IN[1]) == 2
	dim0, dim1 = LAYER_IN[1]
	
	_ntm_module.relu_dlayer_in(LAYER_IN[0], LAYER_IN[1], OUT_BUFFER[0], thresh, gpu_ind)
	OUT_BUFFER[1] = (dim0, dim1, dim0, dim1)

def sigmoid_dlayer_in(LAYER_OUT, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(LAYER_OUT)
	check_buffer(OUT_BUFFER)
	assert len(LAYER_OUT[1]) == 2
	
	dim0, dim1 = LAYER_OUT[1]
	
	_ntm_module.sigmoid_dlayer_in(LAYER_OUT[0], LAYER_OUT[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (dim0, dim1, dim0, dim1)

def focus_key_dkeys(BETA_OUT, KEYS, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(BETA_OUT)
	check_buffer(KEYS)
	check_buffer(OUT_BUFFER)
	assert len(KEYS[1]) == 2
	
	n_controllers, mem_length = KEYS[1]
	
	_ntm_module.focus_key_dkeys(BETA_OUT[0], KEYS[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (n_controllers, mem_length, n_controllers, mem_length)

def focus_key_dbeta_out(KEYS, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(KEYS)
	check_buffer(OUT_BUFFER)
	assert len(KEYS[1]) == 2
	
	n_controllers, mem_length = KEYS[1]

	_ntm_module.focus_key_dbeta_out(KEYS[0], KEYS[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (n_controllers, mem_length, n_controllers, 1)

def sharpen_dgamma(W, GAMMA, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(W)
	check_buffer(GAMMA)
	check_buffer(OUT_BUFFER)
	assert len(GAMMA[1]) == len(W[1]) == 2
	assert GAMMA[1][0] == W[1][0]
	
	dim0, dim1 = W[1]
	
	_ntm_module.sharpen_dgamma(W[0], W[1], GAMMA[0], GAMMA[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (dim0, dim1, dim0, 1)

def sharpen_dw(W, GAMMA, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(W)
	check_buffer(GAMMA)
	check_buffer(OUT_BUFFER)
	assert len(GAMMA[1]) == len(W[1]) == 2
	assert GAMMA[1][0] == W[1][0]
	
	dim0, dim1 = W[1]
	
	_ntm_module.sharpen_dw(W[0], W[1], GAMMA[0], GAMMA[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (dim0, dim1, dim0, dim1)

def softmax_dlayer_in(LAYER_OUT, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(LAYER_OUT)
	check_buffer(OUT_BUFFER)
	assert len(LAYER_OUT[1]) == 2
	
	dim0, dim1 = LAYER_OUT[1]
	
	_ntm_module.softmax_dlayer_in(LAYER_OUT[0], LAYER_OUT[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (dim0, dim1, dim0, dim1)

def cosine_sim_expand_dmem(KEYS, MEM, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(KEYS)
	check_buffer(MEM)
	check_buffer(OUT_BUFFER)
	assert len(KEYS[1]) == len(MEM[1]) == 2
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers = KEYS[1][0]
	mem_length = KEYS[1][1]
	M = MEM[1][0]
	
	_ntm_module.cosine_sim_expand_dmem(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (n_controllers, M, M, mem_length)

def cosine_sim_expand_dkeys(KEYS, MEM, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(KEYS)
	check_buffer(MEM)
	check_buffer(OUT_BUFFER)
	assert len(KEYS[1]) == len(MEM[1]) == 2
	assert KEYS[0] != MEM[0]
	assert OUT_BUFFER[0] != KEYS[0]
	assert OUT_BUFFER[0] != MEM[0]
	assert KEYS[1][1] == MEM[1][1]

	n_controllers = KEYS[1][0]
	mem_length = KEYS[1][1]
	M = MEM[1][0]
	
	_ntm_module.cosine_sim_expand_dkeys(KEYS[0], KEYS[1], MEM[0], MEM[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (n_controllers, M, n_controllers, mem_length)

def sync(gpu_ind=0):
	assert isinstance(gpu_ind,int)
	return _ntm_module.sync(gpu_ind)

def set_buffer(data, buffer_ind, gpu_ind=0, warn=True):
	assert data.dtype == np.dtype('float32')
	assert isinstance(gpu_ind,int)
	assert isinstance(buffer_ind,int)
	
	if not data.flags.contiguous and warn:
		print 'warning: input to set_buffer not C-contiguous (data)'
		data = np.ascontiguousarray(data)

	return _ntm_module.set_buffer(data, buffer_ind, gpu_ind)

def free_buffer(buffer_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(buffer_ind,int)
	return _ntm_module.free_buffer(buffer_ind, gpu_ind)

def free_all_buffers(gpu_ind=0):
	assert isinstance(gpu_ind,int)
	for buffer_ind in range(n_vars_allocated[gpu_ind]):
		_ntm_module.free_buffer(buffer_ind, gpu_ind)
	n_vars_allocated[gpu_ind] = 0

def return_buffer(BUFFER, warn=1, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(BUFFER)

	return _ntm_module.return_buffer(BUFFER[0], warn, gpu_ind).reshape(BUFFER[1])
	
def dot(BUFFER1, BUFFER2, OUT_BUFFER, increment=0, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(BUFFER1)
	check_buffer(BUFFER2)
	check_buffer(OUT_BUFFER)
	assert len(BUFFER1[1]) == len(BUFFER2[1]) == 2
	assert BUFFER1[1][1] == BUFFER2[1][0]
	assert OUT_BUFFER[0] != BUFFER1[0]
	assert OUT_BUFFER[0] != BUFFER2[0]
	assert (OUT_BUFFER[1] is not None) or increment == 0
	
	_ntm_module.dot(BUFFER1[0], BUFFER1[1], BUFFER2[0], BUFFER2[1], OUT_BUFFER[0], increment, gpu_ind)
	OUT_BUFFER[1] = (BUFFER1[1][0], BUFFER2[1][1])

def add_mem(GW, ADD_OUT, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(GW)
	check_buffer(ADD_OUT)
	check_buffer(OUT_BUFFER)
	assert len(GW[1]) == len(ADD_OUT[1]) == 2
	assert GW[1][0] == ADD_OUT[1][0]
	assert OUT_BUFFER[0] != GW[0]
	assert OUT_BUFFER[0] != ADD_OUT[0]
	
	_ntm_module.add_mem(GW[0], GW[1], ADD_OUT[0], ADD_OUT[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (GW[1][1], ADD_OUT[1][1])

def add_mem_dgw(GW, ADD_OUT, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(GW)
	check_buffer(ADD_OUT)
	check_buffer(OUT_BUFFER)
	assert len(GW[1]) == len(ADD_OUT[1]) == 2
	assert GW[1][0] == ADD_OUT[1][0]
	assert OUT_BUFFER[0] != GW[0]
	assert OUT_BUFFER[0] != ADD_OUT[0]
	
	C,M = GW[1]
	mem_length = ADD_OUT[1][1]
	
	_ntm_module.add_mem_dgw(ADD_OUT[0], GW[1], ADD_OUT[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (M , mem_length, C, M)
	
def add_mem_dadd_out(GW, ADD_OUT, OUT_BUFFER, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	check_buffer(GW)
	check_buffer(ADD_OUT)
	check_buffer(OUT_BUFFER)
	assert len(GW[1]) == len(ADD_OUT[1]) == 2
	assert GW[1][0] == ADD_OUT[1][0]
	assert OUT_BUFFER[0] != GW[0]
	assert OUT_BUFFER[0] != ADD_OUT[0]
	
	C,M = GW[1]
	mem_length = ADD_OUT[1][1]
	
	_ntm_module.add_mem_dadd_out(GW[0], GW[1], ADD_OUT[1], OUT_BUFFER[0], gpu_ind)
	OUT_BUFFER[1] = (M , mem_length, C, mem_length)

def sharpen_dgamma_cpu(w, gamma, warn=True):
	assert w.dtype == np.dtype('float32')
	assert gamma.dtype == np.dtype('float32')
	assert w.ndim == gamma.ndim == 2
	assert gamma.shape[0] == w.shape[0]
	assert gamma.shape[1] == 1

	if not w.flags.contiguous and warn:
		print 'warning: input to sharpen_dw_cpu not C-contiguous (w)'
		w = np.ascontiguousarray(w)
	
	if not gamma.flags.contiguous and warn:
		print 'warning: input to sharpen_dw_cpu not C-contiguous (gamma)'
		gamma = np.ascontiguousarray(gamma)
	
	return _ntm_module.sharpen_dgamma_cpu(w, gamma)

def sharpen_dw_cpu(w, gamma, warn=True):
	assert w.dtype == np.dtype('float32')
	assert gamma.dtype == np.dtype('float32')
	assert w.ndim == gamma.ndim == 2
	assert gamma.shape[0] == w.shape[0]
	assert gamma.shape[1] == 1

	if not w.flags.contiguous and warn:
		print 'warning: input to sharpen_dw_cpu not C-contiguous (w)'
		w = np.ascontiguousarray(w)
	
	if not gamma.flags.contiguous and warn:
		print 'warning: input to sharpen_dw_cpu not C-contiguous (gamma)'
		gamma = np.ascontiguousarray(gamma)

	return _ntm_module.sharpen_dw_cpu(w, gamma)
	
def softmax_dlayer_in_cpu(layer_out, warn=True):
	assert layer_out.dtype == np.dtype('float32')
	assert layer_out.ndim == 2
	
	if not layer_out.flags.contiguous and warn:
		print 'warning: input to softmax_dlayer_in_cpu not C-contiguous'
		layer_out = np.ascontiguousarray(layer_out)
		
	return _ntm_module.softmax_dlayer_in_cpu(layer_out)
	
def cosine_sim_expand_dmem_cpu(keys, mem, warn=True):
	assert keys.dtype == np.dtype('float32')
	assert mem.dtype == np.dtype('float32')
	assert keys.ndim == mem.ndim == 2
	assert keys.shape[1] == mem.shape[1]
	
	if not keys.flags.contiguous and warn:
		print 'warning: input to cosine_sim_expand_dkeys_cpu not C-contiguous (keys)'
		keys = np.ascontiguousarray(keys)
		
	if not mem.flags.contiguous and warn:
		print 'warning: input to cosine_sim_expand_dkeys_cpu not C-contiguous (mem)'
		mem = np.ascontiguousarray(mem)

	return _ntm_module.cosine_sim_expand_dmem_cpu(keys, mem)
	
def cosine_sim_expand_dkeys_cpu(keys, mem, warn=True):
	assert keys.dtype == np.dtype('float32')
	assert mem.dtype == np.dtype('float32')
	assert keys.ndim == mem.ndim == 2
	assert keys.shape[1] == mem.shape[1]
	
	if not keys.flags.contiguous and warn:
		print 'warning: input to cosine_sim_expand_dkeys_cpu not C-contiguous (keys)'
		keys = np.ascontiguousarray(keys)
		
	if not mem.flags.contiguous and warn:
		print 'warning: input to cosine_sim_expand_dkeys_cpu not C-contiguous (mem)'
		mem = np.ascontiguousarray(mem)

	return _ntm_module.cosine_sim_expand_dkeys_cpu(keys, mem)
	

############################################################
def init_buffer(DATA=None, gpu_ind=0):
	if DATA is not None:
		DATA_G = [n_vars_allocated[gpu_ind], DATA.shape]
		set_buffer(DATA, DATA_G[0])
	else:
		DATA_G = [n_vars_allocated[gpu_ind], None]
	n_vars_allocated[gpu_ind] += 1
	return DATA_G

def set_list_buffer(DATA):
	LIST = [None]*len(DATA)
	
	for i in range(len(DATA)):
		LIST[i] = init_buffer(DATA[i])
	return LIST

def return_list_buffer(LIST, SHAPE=None, warn=0):
	DATA = [None]*len(LIST)
	for i in range(len(LIST)):
		try:
			DATA[i] = return_buffer(LIST[i], warn=warn)
		except:
			j = 1
		if SHAPE is not None:
			DATA[i] = DATA[i].reshape(SHAPE[i][1])
	return DATA
	
def print_n_vars_allocated():
	print n_vars_allocated[0]
