import _ntm_module
import numpy as np

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

def return_buffer(buffer_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(buffer_ind,int)

	return _ntm_module.return_buffer(buffer_ind, gpu_ind)
	
def dot(buffer_ind1, buffer_shape1, buffer_ind2, buffer_shape2, out_buffer_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(buffer_ind1,int)
	assert isinstance(buffer_ind2,int)
	assert isinstance(out_buffer_ind,int)
	assert isinstance(buffer_shape1, tuple)
	assert isinstance(buffer_shape2, tuple)
	assert len(buffer_shape1) == len(buffer_shape2) == 2
	assert buffer_shape1[1] == buffer_shape2[0]
	assert out_buffer_ind != buffer_ind1
	assert out_buffer_ind != buffer_ind2
	
	return _ntm_module.dot(buffer_ind1, buffer_shape1, buffer_ind2, buffer_shape2, out_buffer_ind, gpu_ind)
