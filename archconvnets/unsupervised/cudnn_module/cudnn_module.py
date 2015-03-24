import _cudnn_module
import time
import numpy as np

N_BUFFERS = 512
_cudnn_module.init_buffers(N_BUFFERS,N_BUFFERS,N_BUFFERS) # setup descriptors
conv_filter_ind = np.zeros(N_BUFFERS,dtype='int')
conv_img_ind = np.zeros(N_BUFFERS,dtype='int')
n_imgs_buffer = np.zeros(N_BUFFERS,dtype='int')
n_filters_buffer = np.zeros(N_BUFFERS,dtype='int')

def unpool(conv_output, output_switches_x, output_switches_y, img_sz, warn=True):
	assert conv_output.shape[-1] == conv_output.shape[-2]
	assert output_switches_y.shape[0] == output_switches_x.shape[0] == conv_output.shape[0]
	assert output_switches_y.shape[1] == output_switches_x.shape[1] == conv_output.shape[1]
	assert output_switches_y.shape[2] == output_switches_x.shape[2] == conv_output.shape[2]
	assert output_switches_y.shape[3] == output_switches_x.shape[3] == conv_output.shape[3]
	assert conv_output.dtype == np.dtype('float32')
	assert isinstance(img_sz,int)
	
	if not conv_output.flags.contiguous and warn:
		print 'warning: input to unpool not C-contiguous (conv_output)'
		conv_output = np.ascontiguousarray(conv_output)
	if not output_switches_x.flags.contiguous and warn:
		print 'warning: input to unpool not C-contiguous (output_switches_x)'
		output_switches_x = np.ascontiguousarray(output_switches_x)
	if not output_switches_y.flags.contiguous and warn:
		print 'warning: input to unpool not C-contiguous (output_switches_y)'
		output_switches_y = np.ascontiguousarray(output_switches_y)
	
	return _cudnn_module.unpool(conv_output, output_switches_x, output_switches_y, img_sz)

def max_pool_cudnn(conv_output,gpu=0,warn=True):
	assert conv_output.shape[-1] == conv_output.shape[-2]
	assert conv_output.dtype == np.dtype('float32')
	assert isinstance(gpu,int)
	
	if not conv_output.flags.contiguous and warn:
		print 'warning: input to max_pool_cudnn not C-contiguous (conv_output)'
		conv_output = np.ascontiguousarray(conv_output)
	
	return _cudnn_module.max_pool_cudnn(conv_output, gpu)

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

def conv_ddata(filters, imgs, conv_out, PAD=0, gpu_ind=0, warn=True):
	n_filters, n_channels, filter_sz, filter_sz2  = filters.shape
	n_imgs, n_channels2, img_sz, img_sz2 = imgs.shape
	assert imgs.shape[0] == conv_out.shape[0]
	assert isinstance(PAD,int)
	assert isinstance(gpu_ind,int)
	
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
	
	return _cudnn_module.conv_ddata(filters, imgs, conv_out, PAD, gpu_ind)

def conv_dfilter_stream(filters, imgs, conv_out, PAD=0,gpu=0):
	n_filters, n_channels, filter_sz, filter_sz2  = filters.shape
	n_imgs, n_channels2, img_sz, img_sz2 = imgs.shape
	assert imgs.shape[0] == conv_out.shape[0]
	
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert type(filters) == np.ndarray and type(imgs) == np.ndarray
	assert filters.dtype == np.dtype('float32') and imgs.dtype == np.dtype('float32')
	assert conv_out.dtype == np.dtype('float32')
	assert isinstance(gpu,int)
	assert isinstance(PAD,int)
	
	if not imgs.flags.contiguous:
		print 'warning: input to conv not C-contiguous (imgs)'
		imgs=np.ascontiguousarray(imgs)
	if not filters.flags.contiguous:
		print 'warning: input to conv not C-contiguous (filters)'
		filters=np.ascontiguousarray(filters)
	if not conv_out.flags.contiguous:
		print 'warning: input to conv not C-contiguous (conv_out)'
		conv_out=np.ascontiguousarray(conv_out)
	
	return _cudnn_module.conv_dfilter_stream(filters, imgs, conv_out, PAD, gpu)

def conv_dfilter(filters, imgs, conv_out, PAD=0,gpu=0,warn=True):
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
def conv(filters, imgs, PAD=0, gpu=0,warn=True):
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

# set filter buffer on GPU, used by conv_from_buffers()
# inputs: buff_ind, filters [n_filters, channels, filter_sz, filter_sz]
def set_filter_buffer(buff_ind, filters):
	n_filters, n_channels, filter_sz, filter_sz2 = filters.shape
	
	assert filter_sz == filter_sz2
	assert type(filters) == np.ndarray
	assert filters.dtype == np.dtype('float32')
	
	if not filters.flags.contiguous:
		print 'warning: input to set_filter_buffer not C-contiguous (filters)'
		filters=np.ascontiguousarray(filters)
	
	n_filters_buffer[buff_ind] = n_filters
	return _cudnn_module.set_filter_buffer(buff_ind, filters)
	
# set img buffer on GPU, used by conv_from_buffers()
# inputs: buff_ind, imgs [n_imgs, channels, img_sz, img_sz]
def set_img_buffer(buff_ind, imgs):
	n_imgs, n_channels, img_sz, img_sz2 = imgs.shape
	
	assert img_sz == img_sz2
	assert type(imgs) == np.ndarray
	assert imgs.dtype == np.dtype('float32')
	
	if not imgs.flags.contiguous:
		print 'warning: input to set_img_buffer not C-contiguous (imgs)'
		imgs=np.ascontiguousarray(imgs)
	
	n_imgs_buffer[buff_ind] = n_imgs
	return _cudnn_module.set_img_buffer(buff_ind, imgs)

# create output buffer
def set_conv_buffer(conv_buff_ind, filter_buff_ind, img_buff_ind):
	conv_filter_ind[conv_buff_ind] = filter_buff_ind
	conv_img_ind[conv_buff_ind] = img_buff_ind
	return _cudnn_module.set_conv_buffer(conv_buff_ind, filter_buff_ind, img_buff_ind)
	
# convolve based on buffers
# first run set_filter_buffer() and set_img_buffer(), then run set_conv_buffer()
def conv_from_buffers(conv_buff_ind):
	n_imgs = n_imgs_buffer[conv_img_ind[conv_buff_ind]]
	n_filters = n_filters_buffer[conv_filter_ind[conv_buff_ind]]
	out = _cudnn_module.conv_from_buffers(conv_buff_ind)
	conv_out_sz = np.sqrt(out.shape[0]/(n_imgs*n_filters))
	return out.reshape((n_imgs, n_filters, conv_out_sz, conv_out_sz))

# conv_output: [n_imgs, n_sets, n_filters, conv_output_sz, conv_output_sz]
# output_switches_x: [n_imgs, n_filters, output_sz, output_sz]
def max_pool_locs_alt(conv_output, output_switches_x, output_switches_y,warn=True):
	assert conv_output.shape[-1] == conv_output.shape[-2]
	assert conv_output.shape[2] == output_switches_x.shape[1]
	assert conv_output.shape[0] == output_switches_x.shape[0]
	assert output_switches_y.shape[0] == output_switches_x.shape[0]
	assert output_switches_y.shape[1] == output_switches_x.shape[1]
	assert output_switches_y.shape[2] == output_switches_x.shape[2]
	assert output_switches_y.shape[3] == output_switches_x.shape[3]
	assert conv_output.dtype == np.dtype('float32')
	if not conv_output.flags.contiguous and warn:
		print 'warning: input to max_pool_locs_alt not C-contiguous (conv_output)'
		conv_output = np.ascontiguousarray(conv_output)
	if not output_switches_x.flags.contiguous and warn:
		print 'warning: input to max_pool_locs_alt not C-contiguous (output_switches_x)'
		output_switches_x = np.ascontiguousarray(output_switches_x)
	if not output_switches_y.flags.contiguous and warn:
		print 'warning: input to max_pool_locs_alt not C-contiguous (output_switches_y)'
		output_switches_y = np.ascontiguousarray(output_switches_y)
	
	z = _cudnn_module.max_pool_locs_alt(conv_output, output_switches_x, output_switches_y)

	n_filters = conv_output.shape[2]
	n_imgs = conv_output.shape[0]
	n_sets = conv_output.shape[1]
	output_sz = output_switches_x.shape[2]

	return np.ascontiguousarray(z.reshape((n_imgs, n_sets, n_filters, output_sz, output_sz)))

# conv_output: [n_imgs, n_filters, conv_output_sz, conv_output_sz]
# output_switches_x: [n_imgs, n_filters, output_sz, output_sz]
# imgs: [n_imgs, n_channels, img_sz, img_sz]
def max_pool_locs_alt_patches(conv_output, output_switches_x, output_switches_y, imgs, s):
	assert conv_output.shape[-1] == conv_output.shape[-2]
	assert conv_output.shape[1] == output_switches_x.shape[1]
	assert conv_output.shape[0] == output_switches_x.shape[0] == imgs.shape[0]
	assert output_switches_y.shape[0] == output_switches_x.shape[0]
	assert output_switches_y.shape[1] == output_switches_x.shape[1]
	assert output_switches_y.shape[2] == output_switches_x.shape[2]
	assert output_switches_y.shape[3] == output_switches_x.shape[3]
	assert conv_output.dtype == np.dtype('float32')
	assert imgs.dtype == np.dtype('float32')
	if not imgs.flags.contiguous:
		print 'warning: input to max_pool_locs_alt_patches not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	if not conv_output.flags.contiguous:
		print 'warning: input to max_pool_locs_alt_patches not C-contiguous (conv_output)'
		conv_output = np.ascontiguousarray(conv_output)
	if not output_switches_x.flags.contiguous:
		print 'warning: input to max_pool_locs_alt_patches not C-contiguous (output_switches_x)'
		output_switches_x = np.ascontiguousarray(output_switches_x)
	if not output_switches_y.flags.contiguous:
		print 'warning: input to max_pool_locs_alt_patches not C-contiguous (output_switches_y)'
		output_switches_y = np.ascontiguousarray(output_switches_y)
	
	z,y = _cudnn_module.max_pool_locs_alt_patches(conv_output, output_switches_x, output_switches_y, imgs, s)

	n_filters = conv_output.shape[1]
	n_imgs = conv_output.shape[0]
	conv_output_sz = conv_output.shape[2]
	output_sz = output_switches_x.shape[2]
	n_channels = imgs.shape[1]

	return np.ascontiguousarray(z.reshape((n_imgs, n_filters, output_sz, output_sz))), np.ascontiguousarray(y.reshape((n_imgs, n_channels, s, s, n_filters, output_sz, output_sz)))
