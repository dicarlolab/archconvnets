import _ntm_module
import numpy as np

def set_buffer(data, buffer_ind, gpu_ind=0, warn=True):
	assert data.dtype == np.dtype('float32')
	assert isinstance(gpu_ind,int)
	assert isinstance(buffer_ind,int)
	
	if not data.flags.contiguous and warn:
		print 'warning: input to set_2d_buffer not C-contiguous (data)'
		data = np.ascontiguousarray(data)

	return _ntm_module.set_buffer(data, buffer_ind, gpu_ind)
	
def dot_cpu(buffer_ind1, buffer_shape1, buffer_ind2, buffer_shape2, gpu_ind=0, warn=True):
	assert isinstance(gpu_ind,int)
	assert isinstance(buffer_ind1,int)
	assert isinstance(buffer_ind2,int)
	assert isinstance(buffer_shape1, tuple)
	assert isinstance(buffer_shape2, tuple)
	assert len(buffer_shape1) == len(buffer_shape2) == 2
	assert buffer_shape1[1] == buffer_shape2[0]
	
	return _ntm_module.dot_cpu(buffer_ind1, buffer_shape1, buffer_ind2, buffer_shape2, gpu_ind)

def dot_gpu(buffer_ind1, buffer_shape1, buffer_ind2, buffer_shape2, out_buffer_ind, gpu_ind=0, warn=True):
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
	
	# setup output buffer
	set_buffer(np.zeros((buffer_shape1[0], buffer_shape2[1]), dtype='float32'), out_buffer_ind, gpu_ind=gpu_ind)
	
	return _ntm_module.dot_gpu(buffer_ind1, buffer_shape1, buffer_ind2, buffer_shape2, out_buffer_ind, gpu_ind)
