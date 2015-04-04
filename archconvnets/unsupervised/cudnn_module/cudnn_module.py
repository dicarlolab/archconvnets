import _cudnn_module
import numpy as np

def pred_buffer(FL_ind, max3_ind, out_ind, gpu=0, warn=True):
	assert isinstance(gpu,int)
	assert isinstance(max3_ind,int)
	assert isinstance(out_ind,int)
	
	return _cudnn_module.pred_buffer(FL_ind, max3_ind, out_ind, gpu)

def max_pool_cudnn_buffers(imgs_ind, out_ind, gpu=0):
	assert isinstance(imgs_ind, int)
	assert isinstance(out_ind, int)
	assert isinstance(gpu,int)
	
	return _cudnn_module.max_pool_cudnn_buffers(imgs_ind, out_ind, gpu)

def conv_buffers(filters_ind, imgs_ind, out_ind, PAD=2, gpu=0):
	assert isinstance(PAD,int)
	assert isinstance(gpu,int)
	assert isinstance(filters_ind,int)
	assert isinstance(imgs_ind,int)
	assert isinstance(out_ind,int)
	
	return _cudnn_module.conv_buffers(filters_ind, imgs_ind, out_ind, PAD, gpu)

def conv_ddata_buffers(filters_ind, imgs_ind, conv_out_ind, out_ind, PAD=2, gpu=0):
	assert isinstance(PAD,int)
	assert isinstance(gpu,int)
	assert isinstance(filters_ind,int)
	assert isinstance(imgs_ind,int)
	assert isinstance(conv_out_ind,int)
	assert isinstance(out_ind,int)
	
	return _cudnn_module.conv_ddata_buffers(filters_ind, imgs_ind, conv_out_ind, out_ind, PAD, gpu)

def conv_dfilter_buffers(filters_ind, imgs_ind, conv_out_ind, out_ind, PAD=2,stream=0,gpu=0):
	assert isinstance(gpu,int)
	assert isinstance(PAD,int)
	assert isinstance(filters_ind,int)
	assert isinstance(imgs_ind,int)
	assert isinstance(conv_out_ind,int)
	assert isinstance(out_ind,int)
	assert isinstance(stream,int)

	return _cudnn_module.conv_dfilter_buffers(filters_ind, imgs_ind, conv_out_ind, out_ind, PAD, stream, gpu)

def return_2d_buffer(buffer_ind, gpu=0, stream=-1):
	assert isinstance(gpu,int)
	assert isinstance(buffer_ind,int)
	assert isinstance(stream,int)
	
	return _cudnn_module.return_2d_buffer(buffer_ind, stream, gpu)

def return_buffer(buffer_ind, gpu=0, stream=-1):
	assert isinstance(gpu,int)
	assert isinstance(buffer_ind,int)
	assert isinstance(stream,int)
	
	return _cudnn_module.return_buffer(buffer_ind, stream, gpu)

def set_2d_buffer(data, buffer_ind, gpu=0, warn=True):
	assert len(data.shape) == 2
	
	assert data.dtype == np.dtype('float32')
	assert isinstance(gpu,int)
	assert isinstance(buffer_ind,int)
	
	if not data.flags.contiguous and warn:
		print 'warning: input to set_2d_buffer not C-contiguous (data)'
		data = np.ascontiguousarray(data)

	return _cudnn_module.set_2d_buffer(data, buffer_ind, gpu)

def set_buffer(data, buffer_ind, gpu=0, filter_flag=0, warn=True):
	assert data.shape[-1] == data.shape[-2]
	assert len(data.shape) == 4
	assert isinstance(filter_flag,int)
	assert filter_flag == 0 or filter_flag == 1
	
	assert data.dtype == np.dtype('float32')
	assert isinstance(gpu,int)
	assert isinstance(buffer_ind,int)
	
	if not data.flags.contiguous and warn:
		print 'warning: input to set_buffer not C-contiguous (data)'
		data = np.ascontiguousarray(data)

	return _cudnn_module.set_buffer(data, buffer_ind, filter_flag, gpu)

def max_pool_cudnn(conv_output,gpu=0,warn=True):
	assert conv_output.shape[-1] == conv_output.shape[-2]
	assert conv_output.dtype == np.dtype('float32')
	assert isinstance(gpu,int)
	
	if not conv_output.flags.contiguous and warn:
		print 'warning: input to max_pool_cudnn not C-contiguous (conv_output)'
		conv_output = np.ascontiguousarray(conv_output)
	
	return _cudnn_module.max_pool_cudnn(conv_output, gpu)


def max_pool_back_cudnn_buffers(src_ind, src_diff_ind, dest_ind, out_ind, gpu=0):
	assert isinstance(gpu,int)
	assert isinstance(src_ind,int)
	assert isinstance(src_diff_ind,int)
	assert isinstance(dest_ind,int)
	assert isinstance(out_ind,int)

	return _cudnn_module.max_pool_back_cudnn_buffers(src_ind, src_diff_ind, dest_ind, out_ind, gpu)

def max_pool_back_cudnn(srcData, srcDiffData, destData, gpu=0, warn=True):
	assert srcData.shape[-1] == srcData.shape[-2]
	assert srcDiffData.shape[-1] == srcDiffData.shape[-2]
	assert destData.shape[-1] == destData.shape[-2]
	
	#assert destData.shape[-1] == (srcData.shape[-1]/2) == (srcDiffData.shape[-1]/2)
	
	
	assert srcData.shape[0] == srcDiffData.shape[0] == destData.shape[0]
	assert srcData.shape[1] == srcDiffData.shape[1] == destData.shape[1]
	
	assert srcData.dtype == np.dtype('float32')
	assert srcDiffData.dtype == np.dtype('float32')
	assert destData.dtype == np.dtype('float32')
	assert isinstance(gpu,int)
	
	if not srcData.flags.contiguous and warn:
		print 'warning: input to max_pool_back_cudnn not C-contiguous (srcData)'
		srcData = np.ascontiguousarray(srcData)
	if not srcDiffData.flags.contiguous and warn:
		print 'warning: input to max_pool_back_cudnn not C-contiguous (srcDiffData)'
		srcDiffData = np.ascontiguousarray(srcDiffData)
	if not destData.flags.contiguous and warn:
		print 'warning: input to max_pool_back_cudnn not C-contiguous (destData)'
		destData = np.ascontiguousarray(destData)

	return _cudnn_module.max_pool_back_cudnn(srcData, srcDiffData, destData, gpu)

def conv_ddata(filters, imgs, conv_out, PAD=2, gpu=0, warn=True):
	n_filters, n_channels, filter_sz, filter_sz2  = filters.shape
	n_imgs, n_channels2, img_sz, img_sz2 = imgs.shape
	assert imgs.shape[0] == conv_out.shape[0]
	assert isinstance(PAD,int)
	assert isinstance(gpu,int)
	
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert type(filters) == np.ndarray and type(imgs) == np.ndarray
	assert filters.dtype == np.dtype('float32') and imgs.dtype == np.dtype('float32')
	assert conv_out.dtype == np.dtype('float32')
	
	if not imgs.flags.contiguous and warn == True:
		print 'warning: input to conv not C-contiguous (imgs)'
		imgs=np.ascontiguousarray(imgs)
	if not filters.flags.contiguous and warn == True:
		print 'warning: input to conv not C-contiguous (filters)'
		filters=np.ascontiguousarray(filters)
	if not conv_out.flags.contiguous and warn == True:
		print 'warning: input to conv not C-contiguous (conv_out)'
		conv_out=np.ascontiguousarray(conv_out)
	
	return _cudnn_module.conv_ddata(filters, imgs, conv_out, PAD, gpu)

def conv_dfilter(filters, imgs, conv_out, PAD=2,gpu=0,warn=True):
	n_filters, n_channels, filter_sz, filter_sz2  = filters.shape
	n_imgs, n_channels2, img_sz, img_sz2 = imgs.shape
	assert imgs.shape[0] == conv_out.shape[0]
	assert isinstance(gpu,int)
	assert isinstance(PAD,int)
	
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert type(filters) == np.ndarray and type(imgs) == np.ndarray
	assert filters.dtype == np.dtype('float32') and imgs.dtype == np.dtype('float32')
	assert conv_out.dtype == np.dtype('float32')
	
	if not imgs.flags.contiguous and warn == True:
		print 'warning: input to conv not C-contiguous (imgs)'
		imgs=np.ascontiguousarray(imgs)
	if not filters.flags.contiguous and warn == True:
		print 'warning: input to conv not C-contiguous (filters)'
		filters=np.ascontiguousarray(filters)
	if not conv_out.flags.contiguous and warn == True:
		print 'warning: input to conv not C-contiguous (conv_out)'
		conv_out=np.ascontiguousarray(conv_out)
	
	return _cudnn_module.conv_dfilter(filters, imgs, conv_out, PAD, gpu)

# standard convolution, requires no pre-set buffers.
# inputs: filters [n_filters, channels, filter_sz, filter_sz]
#			imgs [n_imgs, channels, img_sz, img_sz]
def conv(filters, imgs, PAD=2, gpu=0,warn=True):
	n_filters, n_channels, filter_sz, filter_sz2  = filters.shape
	n_imgs, n_channels2, img_sz, img_sz2 = imgs.shape
	assert isinstance(gpu,int)
	assert isinstance(PAD,int)
	assert PAD >= 0
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert type(filters) == np.ndarray and type(imgs) == np.ndarray
	assert filters.dtype == np.dtype('float32') and imgs.dtype == np.dtype('float32')
	
	if not imgs.flags.contiguous and warn:
		print 'warning: input to conv not C-contiguous (imgs)'
		imgs=np.ascontiguousarray(imgs)
	if not filters.flags.contiguous and warn:
		print 'warning: input to conv not C-contiguous (filters)'
		filters=np.ascontiguousarray(filters)
	
	return _cudnn_module.conv(filters, imgs, PAD, gpu)
