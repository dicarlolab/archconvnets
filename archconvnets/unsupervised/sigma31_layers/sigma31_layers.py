import _sigma31_layers
import time
import numpy as np

# output_switches3_x, output_switches3_y, [n_imgs, n3, max_output_sz3, max_output_sz3]
# output_switches2_x, output_switches2_y, [n_imgs, n2, max_output_sz2, max_output_sz2]
# output_switches1_x, output_switches1_y, [n_imgs, n1, max_output_sz1, max_output_sz1]
# ints: s1, s2, s3
# labels [n_imgs]
# imgs: [n_imgs, 3, img_sz, img_sz] (float32)
# int: N_C
def s31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs, N_C):
	assert isinstance(s1,int)
	assert isinstance(s2,int)
	assert isinstance(s3,int)
	assert isinstance(N_C,int)
	assert output_switches1_x.shape[0] == output_switches2_x.shape[0] == output_switches3_x.shape[0] == imgs.shape[0] == labels.shape[0]
	assert output_switches1_x.shape[2] == output_switches1_y.shape[3]
	assert output_switches2_x.shape[2] == output_switches2_y.shape[3]
	assert output_switches3_x.shape[2] == output_switches3_y.shape[3]
	assert imgs.shape[2] == imgs.shape[3]
	
	N_IMGS = imgs.shape[0]
	n3 = output_switches3_x.shape[1]
	n2 = output_switches2_x.shape[1]
	n1 = output_switches1_x.shape[1]
	max_output_sz3 = output_switches3_x.shape[2]
	
	if not imgs.flags.contiguous:
		print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	
	sigma31_L1, sigma31_L2, sigma31_L3, sigma31_FL = _sigma31_layers.compute_sigma31_reduced(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs, N_C)
	

	sigma31_L1 /= (N_IMGS*(s2**2)*(s3**2)*(max_output_sz3**2)*n2*n3)
	sigma31_L2 /= (N_IMGS*(s1**2)*(s3**2)*(max_output_sz3**2)*3*n3)
	sigma31_L3 /= (N_IMGS*(s1**2)*(s2**2)*(max_output_sz3**2)*3*n1)
	sigma31_FL /= (N_IMGS*(s1**2)*(s2**2)*(s3**2)*3*n1*n2)
	
	sigma31_L1 = sigma31_L1.reshape((N_C, 3, n1, s1, s1))
	sigma31_L2 = sigma31_L2.reshape((N_C, n2, n1, s2, s2))
	sigma31_L3 = sigma31_L3.reshape((N_C, n3, n2, s3, s3))
	sigma31_FL = sigma31_FL.reshape((N_C, n3, max_output_sz3, max_output_sz3))
	return sigma31_L1, sigma31_L2, sigma31_L3, sigma31_FL


	
# output_switches3_x, output_switches3_y, [n_imgs, n3, max_output_sz3, max_output_sz3]
# output_switches2_x, output_switches2_y, [n_imgs, n2, max_output_sz2, max_output_sz2]
# output_switches1_x, output_switches1_y, [n_imgs, n1, max_output_sz1, max_output_sz1]
# ints: s1, s2, s3
# labels [n_imgs]
# imgs: [n_imgs, 3, img_sz, img_sz] (float32)
# int: N_C
def s31_full(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs, N_C):
	assert isinstance(s1,int)
	assert isinstance(s2,int)
	assert isinstance(s3,int)
	assert isinstance(N_C,int)
	assert output_switches1_x.shape[0] == output_switches2_x.shape[0] == output_switches3_x.shape[0] == imgs.shape[0] == labels.shape[0]
	assert output_switches1_x.shape[2] == output_switches1_y.shape[3]
	assert output_switches2_x.shape[2] == output_switches2_y.shape[3]
	assert output_switches3_x.shape[2] == output_switches3_y.shape[3]
	assert imgs.shape[2] == imgs.shape[3]
	
	N_IMGS = imgs.shape[0]
	n3 = output_switches3_x.shape[1]
	n2 = output_switches2_x.shape[1]
	n1 = output_switches1_x.shape[1]
	max_output_sz3 = output_switches3_x.shape[2]
	
	if not imgs.flags.contiguous:
		print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	
	sigma31 = _sigma31_layers.compute_sigma31_full(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs, N_C)
	

	#sigma31 /= (N_IMGS*(s1**2)*(s2**2)*(s3**2)*(max_output_sz3**2)*n1*n2*n3)
	
	sigma31 = sigma31.reshape((N_C, 3, n1, s1, s1, n2, s2, s2, n3, s3, s3, max_output_sz3, max_output_sz3))
	return sigma31

def einsum_deriv_gpu(layer_ind, deriv_ind, gpu_ind):
	assert isinstance(gpu_ind, int)
	assert isinstance(layer_ind, int)
	assert isinstance(deriv_ind, int)
	
	return _sigma31_layers.einsum_deriv_gpu(layer_ind, deriv_ind, gpu_ind)

def einsum_return(layer_ind, deriv_ind, gpu_ind):
	assert isinstance(gpu_ind, int)
	assert isinstance(layer_ind, int)
	assert isinstance(deriv_ind, int)
	
	return _sigma31_layers.einsum_return(layer_ind, deriv_ind, gpu_ind)

def set_sigma_buffer(sigma31, layer_ind, gpu_ind):
	assert isinstance(layer_ind,int)
	assert isinstance(gpu_ind,int)
	
	assert sigma31.dtype == np.dtype('float32')
	
	if not sigma31.flags.contiguous:
		print 'warning: input not C-contiguous (sigma31)'
		sigma31 = np.ascontiguousarray(sigma31)
	
	return _sigma31_layers.set_sigma_buffer(sigma31, layer_ind, gpu_ind)

def set_filter_buffers(F1, F2, F3, FL, gpu_ind):
	assert isinstance(gpu_ind,int)
	
	assert F1.dtype == np.dtype('float32')
	assert F2.dtype == np.dtype('float32')
	assert F3.dtype == np.dtype('float32')
	assert FL.dtype == np.dtype('float32')
	assert FL.shape[-1] == FL.shape[-2]
	assert F3.shape[-1] == F3.shape[-2]
	assert F2.shape[-1] == F2.shape[-2]
	assert F1.shape[-1] == F1.shape[-2]
	
	if not FL.flags.contiguous:
		print 'warning: input not C-contiguous (FL)'
		FL = np.ascontiguousarray(FL)
	if not F3.flags.contiguous:
		print 'warning: input not C-contiguous (F3)'
		F3 = np.ascontiguousarray(F3)
	if not F2.flags.contiguous:
		print 'warning: input not C-contiguous (F2)'
		F2 = np.ascontiguousarray(F2)
	if not F1.flags.contiguous:
		print 'warning: input not C-contiguous (F1)'
		F1 = np.ascontiguousarray(F1)
	
	return _sigma31_layers.set_filter_buffers(F1, F2, F3, FL, gpu_ind)
