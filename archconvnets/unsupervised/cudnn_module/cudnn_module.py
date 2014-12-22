import _cudnn_module
import time
import numpy as np


def conv(filters, imgs):
	n_channels, filter_sz, filter_sz, n_filters = filters.shape
	n_channels, img_sz, img_sz, n_imgs = imgs.shape
	
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


# #### Run code ##################################################
if __name__ == '__main__':
	n_imgs = 256;
	n_channels = 32;
	img_sz = 64;
	n_filters = 256;
	filter_sz = 5;
	
	x = np.load('/export/batch_storage2/batch128_img138_full/data_batch_3')['data']
	
	np.random.seed(666)
	filters = np.single(np.random.random((n_channels, filter_sz, filter_sz, n_filters)))
	imgs = np.single(np.random.random((n_channels, img_sz, img_sz, n_imgs))) #x[:,8].reshape((3,138,138,1)))
	
	t_start = time.time()
	z = conv(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L1(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L1(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L1(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L2(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L2(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L2(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L3(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L3(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
	
	t_start = time.time()
	z = conv_L3(filters, imgs).reshape((n_imgs, n_filters, 60,60)).transpose((1,2,3,0))
	print time.time() - t_start
