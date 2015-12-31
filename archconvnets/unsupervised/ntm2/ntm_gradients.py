import numpy as np


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
	return numer / denom # [n_controllers, n_mem_slots]

def add_points(args):
	assert len(args) == 2
	assert args[0].shape == args[1].shape
	return args[0] + args[1]

def add_points_dinput(args):
	assert len(args) == 2
	assert args[0].shape == args[1].shape
	out = np.zeros(np.concatenate((args[0].shape, args[0].shape)),dtype='single')
	for i in range(out.shape[0]):
		out[i,range(out.shape[1]),i,range(out.shape[1])] = 1
	return out

def sum_points(args):
	assert len(args) == 1
	return args[0].sum()[np.newaxis]

def sum_points_dinput(args):
	assert len(args) == 1
	return np.ones(tuple(np.concatenate(((1,), args[0].shape))))

def sq_points(args):
	assert len(args) == 1
	input = args[0]
	return input**2

def sq_points_dinput(args):
	input = args[0]
	n = input.shape[1]
	dinput = np.zeros((input.shape[0], n, input.shape[0], n),dtype='single')
	for i in range(input.shape[0]):
		dinput[i,range(n),i,range(n)] = 2*input[i]
	return dinput

def linear_F(args):
	F, layer_in = args
	# F: [n1, n_in], layer_in: [n_in, 1]
	
	return np.dot(F,layer_in) # [n1, 1]

def linear_F_dx(args):
	F, x = args
	n = x.shape[1]
	temp = np.zeros((F.shape[0], n, x.shape[0], n),dtype='single')
	temp[:,range(n),:,range(n)] = F
	return temp

def linear_F_dF(args):
	F, x = args
	n = F.shape[0]
	temp = np.zeros((n, x.shape[1], n, F.shape[1]),dtype='single')
	temp[range(n),:,range(n)] = x.T
	return temp
