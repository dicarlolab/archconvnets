import _cudnn_module
import time
import numpy as np

N_BUFFERS = 512
_cudnn_module.init_buffers(N_BUFFERS,N_BUFFERS,N_BUFFERS) # setup descriptors
conv_filter_ind = np.zeros(N_BUFFERS,dtype='int')
conv_img_ind = np.zeros(N_BUFFERS,dtype='int')
n_imgs_buffer = np.zeros(N_BUFFERS,dtype='int')
n_filters_buffer = np.zeros(N_BUFFERS,dtype='int')

# standard convolution, requires no pre-set buffers.
# inputs: filters [channels, filter_sz, filter_sz, n_filters]
#			imgs [channels, img_sz, img_sz, n_imgs]
def conv(filters, imgs):
	n_channels, filter_sz, filter_sz2, n_filters = filters.shape
	n_channels2, img_sz, img_sz2, n_imgs = imgs.shape
	
	if n_channels != n_channels2 or img_sz != img_sz2 or filter_sz != filter_sz2:
		raise 'input dim problem'
	if type(filters) != np.ndarray or type(imgs) != np.ndarray:
		raise 'argument is not *NumPy* array'
	if filters.dtype != np.dtype('float32') or imgs.dtype != np.dtype('float32'):
		raise 'argument is not *Float* NumPy vector'
	
	if not imgs.flags.contiguous:
		imgs=np.array(imgs)
	if not filters.flags.contiguous:
		filters=np.array(filters)
	
	imgs = imgs.transpose((3,0,1,2))
	filters = filters.transpose((3,0,1,2))
	out = _cudnn_module.conv(filters.ravel(), imgs.ravel(), n_channels, filter_sz, n_filters, img_sz, n_imgs)
	conv_out_sz = np.sqrt(out.shape[0]/(n_imgs*n_filters))
	return out.reshape((n_imgs, n_filters, conv_out_sz, conv_out_sz)).transpose((1,2,3,0))

# set filter buffer on GPU, used by conv_from_buffers()
# inputs: buff_ind, filters [channels, filter_sz, filter_sz, n_filters]
def set_filter_buffer(buff_ind, filters):
	n_channels, filter_sz, filter_sz2, n_filters = filters.shape
	
	if filter_sz != filter_sz2:
		raise 'input dim problem'
	if type(filters) != np.ndarray:
		raise 'argument is not *NumPy* array'
	if filters.dtype != np.dtype('float32'):
		raise 'argument is not *Float* NumPy vector'
	
	if not filters.flags.contiguous:
		filters=np.array(filters)
	
	n_filters_buffer[buff_ind] = n_filters
	filters = filters.transpose((3,0,1,2))
	return _cudnn_module.set_filter_buffer(buff_ind, filters.ravel(), n_channels, filter_sz, n_filters)
	
# set img buffer on GPU, used by conv_from_buffers()
# inputs: buff_ind, imgs [channels, img_sz, img_sz, n_imgs]
def set_img_buffer(buff_ind, imgs):
	n_channels, img_sz, img_sz2, n_imgs = imgs.shape
	
	if img_sz != img_sz2:
		raise 'input dim problem'
	if type(imgs) != np.ndarray:
		raise 'argument is not *NumPy* array'
	if imgs.dtype != np.dtype('float32'):
		raise 'argument is not *Float* NumPy vector'
	
	if not imgs.flags.contiguous:
		imgs=np.array(imgs)
	
	n_imgs_buffer[buff_ind] = n_imgs
	imgs = imgs.transpose((3,0,1,2))
	return _cudnn_module.set_img_buffer(buff_ind, imgs.ravel(), n_channels, img_sz, n_imgs)

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
	print out.reshape((n_imgs, n_filters, conv_out_sz, conv_out_sz)).flags
	print out.reshape((n_imgs, n_filters, conv_out_sz, conv_out_sz)).transpose((1,2,3,0)).shape
	return out.reshape((n_imgs, n_filters, conv_out_sz, conv_out_sz)).transpose((1,2,3,0))
	
def max_pool_locs_alt(conv_output, output_switches_x, output_switches_y):
	assert conv_output.shape[1] == conv_output.shape[2]
	assert conv_output.shape[0] == output_switches_x.shape[0]
	assert conv_output.shape[3] == output_switches_x.shape[3]
	assert output_switches_y.shape[0] == output_switches_x.shape[0]
	assert output_switches_y.shape[1] == output_switches_x.shape[1]
	assert output_switches_y.shape[2] == output_switches_x.shape[2]
	assert output_switches_y.shape[3] == output_switches_x.shape[3]
	assert conv_output.dtype == np.dtype('float32')
	'''if not conv_output.flags.contiguous:
		print 'not contig., conv_output'
		conv_output = np.ascontiguousarray(conv_output)
	if not output_switches_x.flags.contiguous:
		output_switches_x = np.ascontiguousarray(output_switches_x)
		print 'not contig., x'
	if not output_switches_y.flags.contiguous:
		output_switches_y = np.ascontiguousarray(output_switches_y)
		print 'not contig., y'
		'''
	
	z = _cudnn_module.max_pool_locs_alt(conv_output, output_switches_x, output_switches_y)
	n_filters = conv_output.shape[0]
	n_imgs = conv_output.shape[3]
	n_sets = conv_output.shape[4]
	output_sz = output_switches_x.shape[1]

	return z.reshape((n_filters, output_sz, output_sz, n_imgs, n_sets))
