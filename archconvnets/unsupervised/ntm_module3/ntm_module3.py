import _ntm_module3
import numpy as np
import copy
from archconvnets.unsupervised.ntm3.gpu_flag import *

DEBUG = False
N_GPUS = 4
N_BUFFERS = 5000
n_vars_allocated = np.zeros((N_GPUS, N_BUFFERS), dtype='bool') # variable slots allocated per gpu

def return_cpu(buffer_ind):
	return CPU_BUFFER[0][buffer_ind]
	
def sync(gpu_ind=GPU_IND):
	return _ntm_module3.sync(gpu_ind)

def return_buffer_sz(buffer_ind, gpu_ind=GPU_IND):
	return _ntm_module3.return_buffer_sz(buffer_ind, gpu_ind)

def check_buffer(BUFFER, gpu_ind=GPU_IND):
	'''assert len(BUFFER) == 2
	assert isinstance(BUFFER[0], int)
	assert BUFFER[0] >= 0
	assert n_vars_allocated[gpu_ind, BUFFER[0]]
	assert isinstance(BUFFER[1], tuple) or BUFFER[1] == None
	if BUFFER[1] is not None:
		assert return_buffer_sz(BUFFER[0], gpu_ind) == np.prod(BUFFER[1]), 'stored size %i did not match actual size %i' % (return_buffer_sz(BUFFER[0], gpu_ind), np.prod(BUFFER[1]))
	else:
		assert return_buffer_sz(BUFFER[0], gpu_ind) == 0, '%i' % return_buffer_sz(BUFFER[0], gpu_ind)'''
	return

def set_device(gpu_ind=GPU_IND):
	_ntm_module3.set_device(gpu_ind)

def free_buffer(BUFFER, gpu_ind=GPU_IND):
	#check_buffer(BUFFER)
	n_vars_allocated[gpu_ind, BUFFER[0]] = False
	_ntm_module3.free_buffer(BUFFER[0], gpu_ind)
	BUFFER = None

def zero_buffer(BUFFER, gpu_ind=GPU_IND):
	_ntm_module3.zero_buffer(BUFFER[0], gpu_ind)

	
def return_n_allocated(gpu_ind=GPU_IND):
	return n_vars_allocated[gpu_ind].sum()

def free_all_buffers():
	for gpu_ind in range(N_GPUS):
		for buffer_ind in np.nonzero(n_vars_allocated[gpu_ind])[0]:
			_ntm_module3.free_buffer(buffer_ind, gpu_ind)
		
		n_vars_allocated[gpu_ind] = False

def free_list(LIST, gpu_ind=GPU_IND):
	for layer_ind in range(len(LIST)):
		if LIST[layer_ind] is not None:
			# list of list, ex. partials
			if isinstance(LIST[layer_ind][0], list):
				for arg in range(len(LIST[layer_ind])):
					free_buffer(LIST[layer_ind][arg], gpu_ind)
			# outputs
			else:
				free_buffer(LIST[layer_ind], gpu_ind)

def zero_list(LIST, gpu_ind=GPU_IND):
	for layer_ind in range(len(LIST)):
		if LIST[layer_ind] is not None:
			# list of list, ex. partials
			if isinstance(LIST[layer_ind][0], list):
				for arg in range(len(LIST[layer_ind])):
					zero_buffer(LIST[layer_ind][arg], gpu_ind)
			# outputs
			else:
				zero_buffer(LIST[layer_ind], gpu_ind)
				
				
def set_buffer(DATA, DATA_G, gpu_ind=GPU_IND, warn=True):
	if isinstance(DATA, int) or isinstance(DATA, np.single):
		DATA = np.asarray(DATA,dtype='float32')[np.newaxis]
	
	assert DATA.dtype == np.dtype('float32'), DATA.dtype
	if not DATA.flags.contiguous and warn:
		assert False, 'warning: input to init_buffer not C-contiguous'
		DATA = np.ascontiguousarray(DATA)
	
	_ntm_module3.set_buffer(DATA, DATA_G[0], gpu_ind)
	
	DATA_G[1] = DATA.shape
	return DATA_G

def init_buffer(DATA=None, gpu_ind=GPU_IND, warn=True):
	z = np.nonzero(1-n_vars_allocated[gpu_ind])[0]
	assert len(z) != 0, 'no memory slots left'
	buffer_ind = z.min()
	if DATA is not None:
		DATA_G = [buffer_ind, DATA.shape]
		DATA_G = set_buffer(DATA, DATA_G, gpu_ind=gpu_ind, warn=warn)
	else:
		DATA_G = [buffer_ind, None]
	n_vars_allocated[gpu_ind,buffer_ind] = True
	return DATA_G

# copy B to A
def copy_buffer(B, A=None, gpu_ind=GPU_IND):
	if A is None:
		A = init_buffer()
	_ntm_module3.copy_buffer(B[0], A[0], gpu_ind)

	A[1] = copy.deepcopy(B[1])
	return A

def copy_list(LIST_B, LIST_A=None, gpu_ind=GPU_IND):
	if LIST_A is None:
		LIST_A = [None]*len(LIST_B)
	
	assert len(LIST_A) == len(LIST_B)
	
	for i in range(len(LIST_B)):
		LIST_A[i] = copy_buffer(LIST_B[i], LIST_A[i], gpu_ind)
	return LIST_A
	
def return_buffer(BUFFER, warn=1, gpu_ind=GPU_IND):
	return _ntm_module3.return_buffer(BUFFER[0], warn, gpu_ind).reshape(BUFFER[1])


# out_buffer = a * scalar0 + b * scalar
# when OUT_BUFFER=None, store results in "a"
t_add = [0]
def point_wise_add(args, OUT_BUFFER=None, scalar=1, scalar0=1, gpu_ind=GPU_IND):
	t = time.time()
	
	A, B = args
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = A[1]
	else:
		OUT_BUFFER = A
	
	_ntm_module3.point_wise_add(A[0], B[0], np.single(scalar), np.single(scalar0), OUT_BUFFER[0], gpu_ind)

	OUT_BUFFER[1] = B[1]
	t_add[0] = time.time() - t
	return OUT_BUFFER

# out_buffer = a / sqrt(b)
# when OUT_BUFFER=None, store results in "a"
def point_wise_div_sqrt(args, OUT_BUFFER=None, clip=10, gpu_ind=GPU_IND):
	A, B = args
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
		OUT_BUFFER[1] = A[1]
	else:
		OUT_BUFFER = A
	
	_ntm_module3.point_wise_div_sqrt(A[0], B[0], OUT_BUFFER[0], np.single(clip), gpu_ind)
	
	OUT_BUFFER[1] = B[1]
	return OUT_BUFFER

# additional_args[0]: Squeeze output or not, 
# additional_args[1] (sum_all): collapse x from [k,j] to [k*j,1] to give an output of [i,1] as opposed to [i,j]
t_dot = [0]
def dot(args, OUT_BUFFER=None, increment=0, additional_args=[True, False], gpu_ind=GPU_IND):
	t = time.time()
	
	squeeze, sum_all = additional_args
	BUFFER1, BUFFER2 = args
	
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer(gpu_ind=gpu_ind)
	
	
	# if source is a conv layer (4D input), sum across everything
	if len(BUFFER2[1]) == 4 or sum_all:
		BUFFER2_reshaped = (np.prod(BUFFER2[1]), 1)
	else:
		BUFFER2_reshaped = BUFFER2[1]
	
	if DEBUG:
		assert len(BUFFER1[1]) >= 2
		assert len(BUFFER2[1]) == 2 or len(BUFFER2[1]) == 4
		assert OUT_BUFFER[0] != BUFFER1[0]
		assert OUT_BUFFER[0] != BUFFER2[0]
		assert (OUT_BUFFER[1] is not None) or increment == 0
		assert BUFFER1[1][-1] == BUFFER2_reshaped[0]
	
	# reshape buffer1 into two dimensions:
	# (a,b,c,d,e) -> (a*b*c*d, e)
	BUFFER1_new_shape = (np.prod(BUFFER1[1][:len(BUFFER1[1])-1]), BUFFER1[1][-1])
	
	_ntm_module3.dot(BUFFER1[0], BUFFER1_new_shape, BUFFER2[0], BUFFER2_reshaped, OUT_BUFFER[0], increment, gpu_ind)
	
	OUT_BUFFER[1] = tuple(np.concatenate((np.asarray(BUFFER1[1][:len(BUFFER1[1])-1]), np.asarray(BUFFER2_reshaped[1])[np.newaxis])))	
	if squeeze and OUT_BUFFER[1][-1] == 1: # squeeze
		OUT_BUFFER[1] = OUT_BUFFER[1][:len(OUT_BUFFER[1])-1]
	
	t_dot[0] += time.time() - t
	return OUT_BUFFER
	
def zero_buffer_list(WEIGHTS, gpu_ind=GPU_IND):
	for layer_ind in range(len(WEIGHTS)):
		for arg in range(len(WEIGHTS[layer_ind])):
			if WEIGHTS[layer_ind][arg] is not None:
				zero_buffer(WEIGHTS[layer_ind][arg], gpu_ind)


def squeeze_dim1(BUFFER, keep_dims):
	if keep_dims == False: # squeeze
		#assert BUFFER[1][0] == 1
		BUFFER[1] = tuple(BUFFER[1][1:])

def free_list_list(LIST, gpu_ind=GPU_IND):
	for i in range(len(LIST)):
		free_list(LIST[i], gpu_ind)

set_device(GPU_IND)

from gradient_functions.cosine_sim import *
from gradient_functions.linear_F import *
from gradient_functions.add_points import * # sum two layers together, preserving dimensionaltiy
from gradient_functions.sum_points import * # sum one layer into a scalar
from gradient_functions.focus_key import *
from gradient_functions.sigmoid import *
from gradient_functions.sharpen import *
from gradient_functions.relu import *
from gradient_functions.shift_w import *
from gradient_functions.interpolate import *
from gradient_functions.softmax import *
from gradient_functions.sq_points import *
from gradient_functions.dotT import *
from gradient_functions.mult_points import *
from gradient_functions.bias import *
from gradient_functions.conv import *
from gradient_functions.max_pool import *
from gradient_functions.pearson import *
