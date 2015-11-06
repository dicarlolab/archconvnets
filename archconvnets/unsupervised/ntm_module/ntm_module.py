import _ntm_module
import numpy as np

def set_buffer(data, buffer_ind, gpu=0, warn=True):
	assert len(data.shape) == 2
	
	assert data.dtype == np.dtype('float32')
	assert isinstance(gpu,int)
	assert isinstance(buffer_ind,int)
	
	if not data.flags.contiguous and warn:
		print 'warning: input to set_2d_buffer not C-contiguous (data)'
		data = np.ascontiguousarray(data)

	return _ntm_module.set_2d_buffer(data, buffer_ind, gpu)

