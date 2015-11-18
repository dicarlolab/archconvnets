import _ntm_module
import numpy as np

def dsharpen_dgamma(w_ind, w_shape, gamma_ind, gamma_shape, out_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(w_ind,int)
	assert isinstance(gamma_ind,int)
	assert isinstance(out_ind,int)
	assert isinstance(w_shape,tuple)
	assert isinstance(gamma_shape,tuple)
	assert len(gamma_shape) == len(w_shape) == 2
	assert gamma_shape[0] == w_shape[0]

	return _ntm_module.dsharpen_dgamma(w_ind, w_shape, gamma_ind, gamma_shape, out_ind, gpu_ind)

def dsharpen_dw(w_ind, w_shape, gamma_ind, gamma_shape, out_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(w_ind,int)
	assert isinstance(gamma_ind,int)
	assert isinstance(out_ind,int)
	assert isinstance(w_shape,tuple)
	assert isinstance(gamma_shape,tuple)
	assert len(gamma_shape) == len(w_shape) == 2
	assert gamma_shape[0] == w_shape[0]
	
	return _ntm_module.dsharpen_dw(w_ind, w_shape, gamma_ind, gamma_shape, out_ind, gpu_ind)

def dsharpen_dgamma_cpu(w, gamma, warn=True):
	assert w.dtype == np.dtype('float32')
	assert gamma.dtype == np.dtype('float32')
	assert w.ndim == gamma.ndim == 2
	assert gamma.shape[0] == w.shape[0]
	assert gamma.shape[1] == 1

	if not w.flags.contiguous and warn:
		print 'warning: input to dsharpen_dw_cpu not C-contiguous (w)'
		w = np.ascontiguousarray(w)
	
	if not gamma.flags.contiguous and warn:
		print 'warning: input to dsharpen_dw_cpu not C-contiguous (gamma)'
		gamma = np.ascontiguousarray(gamma)
	
	return _ntm_module.dsharpen_dgamma_cpu(w, gamma)

def dsharpen_dw_cpu(w, gamma, warn=True):
	assert w.dtype == np.dtype('float32')
	assert gamma.dtype == np.dtype('float32')
	assert w.ndim == gamma.ndim == 2
	assert gamma.shape[0] == w.shape[0]
	assert gamma.shape[1] == 1

	if not w.flags.contiguous and warn:
		print 'warning: input to dsharpen_dw_cpu not C-contiguous (w)'
		w = np.ascontiguousarray(w)
	
	if not gamma.flags.contiguous and warn:
		print 'warning: input to dsharpen_dw_cpu not C-contiguous (gamma)'
		gamma = np.ascontiguousarray(gamma)

	return _ntm_module.dsharpen_dw_cpu(w, gamma)

def softmax_dlayer_in_nsum(layer_out, layer_out_shape, out_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(out_ind,int)
	assert isinstance(layer_out,int)
	assert isinstance(layer_out_shape,tuple)
	assert len(layer_out_shape) == 2
	
	return _ntm_module.softmax_dlayer_in_nsum(layer_out, layer_out_shape, out_ind, gpu_ind)

def softmax_dlayer_in_nsum_cpu(layer_out, warn=True):
	assert layer_out.dtype == np.dtype('float32')
	assert layer_out.ndim == 2
	
	if not layer_out.flags.contiguous and warn:
		print 'warning: input to softmax_dlayer_in_nsum_cpu not C-contiguous'
		layer_out = np.ascontiguousarray(layer_out)
		
	return _ntm_module.softmax_dlayer_in_nsum_cpu(layer_out)

def cosine_sim_expand_dmem(keys_ind, keys_shape, mem_ind, mem_shape, out_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(keys_ind,int)
	assert isinstance(mem_ind,int)
	assert isinstance(out_ind,int)
	assert isinstance(keys_shape, tuple)
	assert isinstance(mem_shape, tuple)
	assert len(keys_shape) == len(mem_shape) == 2
	assert keys_ind != mem_ind
	assert keys_shape[1] == mem_shape[1]

	return _ntm_module.cosine_sim_expand_dmem(keys_ind, keys_shape, mem_ind, mem_shape, out_ind, gpu_ind)

def cosine_sim_expand_dkeys(keys_ind, keys_shape, mem_ind, mem_shape, out_ind, gpu_ind=0):
	assert isinstance(gpu_ind,int)
	assert isinstance(keys_ind,int)
	assert isinstance(mem_ind,int)
	assert isinstance(out_ind,int)
	assert isinstance(keys_shape, tuple)
	assert isinstance(mem_shape, tuple)
	assert len(keys_shape) == len(mem_shape) == 2
	assert keys_ind != mem_ind
	assert keys_shape[1] == mem_shape[1]

	return _ntm_module.cosine_sim_expand_dkeys(keys_ind, keys_shape, mem_ind, mem_shape, out_ind, gpu_ind)

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
