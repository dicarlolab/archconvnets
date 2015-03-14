import _sigma31_layers
import time
import numpy as np

def F_layer_sum_deriv_inds_gpu(F_partial, F1, F2, F3, FL, layer_ind, gpu_ind, warn=True):
	assert isinstance(layer_ind,int)
	assert isinstance(gpu_ind,int)
	assert F1.dtype == np.dtype('float32')
	assert F2.dtype == np.dtype('float32')
	assert F3.dtype == np.dtype('float32')
	assert FL.dtype == np.dtype('float32')
	assert F_partial.dtype == np.dtype('float32')

	assert F1.shape[-1] == F1.shape[-2]
	assert F2.shape[-1] == F2.shape[-2]
	assert F3.shape[-1] == F3.shape[-2]
	assert FL.shape[-1] == FL.shape[-2]
	
	assert F2.shape[1] == F1.shape[0]
	assert F2.shape[1] == F3.shape[0]
	assert FL.shape[1] == F3.shape[0]
	assert FL.shape[0] == F_partial.shape[0]
	
	n3 = F3.shape[0]
	n2 = F2.shape[0]
	n1 = F1.shape[0]
	
	s1 = F1.shape[-1]
	s2 = F2.shape[-1]
	s3 = F3.shape[-1]
	max_output_sz3 = FL.shape[-1]
	
	#assert np.max(inds) < (n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3)
	
	if not F_partial.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F_partial)'
		F_partial = np.ascontiguousarray(F_partial)
	
	if not F1.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F1)'
		F1 = np.ascontiguousarray(F1)
	
	if not F2.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F2)'
		F2 = np.ascontiguousarray(F2)
	
	if not F3.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F3)'
		F3 = np.ascontiguousarray(F3)
	
	if not FL.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (FL)'
		FL = np.ascontiguousarray(FL)
	
	return _sigma31_layers.compute_F_layer_sum_deriv_inds_gpu(F_partial, F1, F2, F3, FL, layer_ind, gpu_ind)


def F_layer_sum_inds(FL321, F1, F2, F3, FL, inds, layer_ind, warn=True):
	assert isinstance(layer_ind,int)
	assert F1.dtype == np.dtype('float32')
	assert F2.dtype == np.dtype('float32')
	assert F3.dtype == np.dtype('float32')
	assert FL.dtype == np.dtype('float32')
	assert FL321.dtype == np.dtype('float32')

	assert inds.dtype == np.dtype('int64')
	assert np.min(inds) >= 0
	assert len(inds.shape) == 1
	assert np.prod(inds.shape) == FL321.shape[1]
	assert FL321.shape[0] == FL.shape[0]
	
	assert F1.shape[-1] == F1.shape[-2]
	assert F2.shape[-1] == F2.shape[-2]
	assert F3.shape[-1] == F3.shape[-2]
	assert FL.shape[-1] == FL.shape[-2]
	
	assert F2.shape[1] == F1.shape[0]
	assert F2.shape[1] == F3.shape[0]
	assert FL.shape[1] == F3.shape[0]
	
	n3 = F3.shape[0]
	n2 = F2.shape[0]
	n1 = F1.shape[0]
	
	s1 = F1.shape[-1]
	s2 = F2.shape[-1]
	s3 = F3.shape[-1]
	max_output_sz3 = FL.shape[-1]
	
	assert np.max(inds) < (n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3)
	
	if not FL321.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (FL321)'
		FL321 = np.ascontiguousarray(FL321)
	
	if not F1.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F1)'
		F1 = np.ascontiguousarray(F1)
	
	if not F2.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F2)'
		F2 = np.ascontiguousarray(F2)
	
	if not F3.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F3)'
		F3 = np.ascontiguousarray(F3)
	
	if not FL.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (FL)'
		FL = np.ascontiguousarray(FL)
	
	if not inds.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (inds)'
		inds = np.ascontiguousarray(inds)
	
	return _sigma31_layers.compute_F_layer_sum_inds(FL321, F1, F2, F3, FL, inds, layer_ind)

def F_prod_inds(F1, F2, F3, FL, inds, warn=True):
	assert F1.dtype == np.dtype('float32')
	assert F2.dtype == np.dtype('float32')
	assert F3.dtype == np.dtype('float32')
	assert FL.dtype == np.dtype('float32')

	assert inds.dtype == np.dtype('int64')
	assert np.min(inds) >= 0
	assert len(inds.shape) == 1
	
	assert F1.shape[-1] == F1.shape[-2]
	assert F2.shape[-1] == F2.shape[-2]
	assert F3.shape[-1] == F3.shape[-2]
	assert FL.shape[-1] == FL.shape[-2]
	
	assert F2.shape[1] == F1.shape[0]
	assert F2.shape[1] == F3.shape[0]
	assert FL.shape[1] == F3.shape[0]
	
	n3 = F3.shape[0]
	n2 = F2.shape[0]
	n1 = F1.shape[0]
	
	s1 = F1.shape[-1]
	s2 = F2.shape[-1]
	s3 = F3.shape[-1]
	max_output_sz3 = FL.shape[-1]
	
	assert np.max(inds) < (n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3)
	
	if not F1.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F1)'
		F1 = np.ascontiguousarray(F1)
	
	if not F2.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F2)'
		F2 = np.ascontiguousarray(F2)
	
	if not F3.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F3)'
		F3 = np.ascontiguousarray(F3)
	
	if not FL.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (FL)'
		FL = np.ascontiguousarray(FL)
	
	if not inds.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (inds)'
		inds = np.ascontiguousarray(inds)
	
	return _sigma31_layers.compute_F_prod_inds(F1, F2, F3, FL, inds)

def set_img_from_patches(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, imgs, inds, patches, warn=True):
	assert output_switches3_x.dtype == np.dtype('int64')
	assert output_switches3_y.dtype == np.dtype('int64')
	assert output_switches2_x.dtype == np.dtype('int64')
	assert output_switches2_y.dtype == np.dtype('int64')
	assert output_switches1_x.dtype == np.dtype('int64')
	assert output_switches1_y.dtype == np.dtype('int64')
	assert inds.dtype == np.dtype('int64')
	assert np.min(inds) >= 0
	assert len(inds.shape) == 1
	assert len(patches.shape) == 2
	assert inds.shape[0] == patches.shape[1]
	
	assert isinstance(s1,int)
	assert isinstance(s2,int)
	assert isinstance(s3,int)
	assert output_switches1_x.shape[0] == output_switches2_x.shape[0] == output_switches3_x.shape[0] == imgs.shape[0] == patches.shape[0]
	assert output_switches1_x.shape[2] == output_switches1_y.shape[3]
	assert output_switches2_x.shape[2] == output_switches2_y.shape[3]
	assert output_switches3_x.shape[2] == output_switches3_y.shape[3]
	assert imgs.shape[2] == imgs.shape[3]
	
	N_IMGS = imgs.shape[0]
	n3 = output_switches3_x.shape[1]
	n2 = output_switches2_x.shape[1]
	n1 = output_switches1_x.shape[1]
	max_output_sz3 = output_switches3_x.shape[2]
	
	assert np.max(inds) < (n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3)
	
	if not patches.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (patches)'
		patches = np.ascontiguousarray(patches)
		
	if not imgs.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	
	if not inds.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (inds)'
		inds = np.ascontiguousarray(inds)
	
	return _sigma31_layers.set_img_from_patches(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, imgs, inds, patches)

def patch_inds_addresses(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, imgs, N_C, inds, warn=True):
	assert output_switches3_x.dtype == np.dtype('int64')
	assert output_switches3_y.dtype == np.dtype('int64')
	assert output_switches2_x.dtype == np.dtype('int64')
	assert output_switches2_y.dtype == np.dtype('int64')
	assert output_switches1_x.dtype == np.dtype('int64')
	assert output_switches1_y.dtype == np.dtype('int64')
	assert inds.dtype == np.dtype('int64')
	assert np.min(inds) >= 0
	assert len(inds.shape) == 1
	
	
	assert isinstance(s1,int)
	assert isinstance(s2,int)
	assert isinstance(s3,int)
	assert isinstance(N_C,int)
	assert output_switches1_x.shape[0] == output_switches2_x.shape[0] == output_switches3_x.shape[0] == imgs.shape[0]
	assert output_switches1_x.shape[2] == output_switches1_y.shape[3]
	assert output_switches2_x.shape[2] == output_switches2_y.shape[3]
	assert output_switches3_x.shape[2] == output_switches3_y.shape[3]
	assert imgs.shape[2] == imgs.shape[3]
	
	N_IMGS = imgs.shape[0]
	n3 = output_switches3_x.shape[1]
	n2 = output_switches2_x.shape[1]
	n1 = output_switches1_x.shape[1]
	max_output_sz3 = output_switches3_x.shape[2]
	
	assert np.max(inds) < (n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3)
	
	if not imgs.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	
	if not inds.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (inds)'
		inds = np.ascontiguousarray(inds)
	
	return _sigma31_layers.compute_patch_inds_addresses(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, imgs, N_C, inds)

def bp_patch_sigma31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x_s31, output_switches3_y_s31, output_switches2_x_s31, output_switches2_y_s31, output_switches1_x_s31, output_switches1_y_s31, imgs, sigma_imgs, deriv_ind, pred, F1, F2, F3, FL, warn=True):
	assert output_switches3_x.dtype == np.dtype('int64')
	assert output_switches3_y.dtype == np.dtype('int64')
	assert output_switches2_x.dtype == np.dtype('int64')
	assert output_switches2_y.dtype == np.dtype('int64')
	assert output_switches1_x.dtype == np.dtype('int64')
	assert output_switches1_y.dtype == np.dtype('int64')
	
	assert output_switches3_x_s31.dtype == np.dtype('int64')
	assert output_switches3_y_s31.dtype == np.dtype('int64')
	assert output_switches2_x_s31.dtype == np.dtype('int64')
	assert output_switches2_y_s31.dtype == np.dtype('int64')
	assert output_switches1_x_s31.dtype == np.dtype('int64')
	assert output_switches1_y_s31.dtype == np.dtype('int64')
	
	assert isinstance(deriv_ind,int)
	
	assert F1.dtype == np.dtype('float32')
	assert F2.dtype == np.dtype('float32')
	assert F3.dtype == np.dtype('float32')
	assert FL.dtype == np.dtype('float32')

	assert F1.shape[-1] == F1.shape[-2]
	assert F2.shape[-1] == F2.shape[-2]
	assert F3.shape[-1] == F3.shape[-2]
	assert FL.shape[-1] == FL.shape[-2]
	
	assert F2.shape[1] == F1.shape[0]
	assert F2.shape[1] == F3.shape[0]
	assert FL.shape[1] == F3.shape[0]
	
	assert output_switches1_x.shape[0] == output_switches2_x.shape[0] == output_switches3_x.shape[0] == imgs.shape[0]
	assert output_switches1_x.shape[2] == output_switches1_y.shape[3]
	assert output_switches2_x.shape[2] == output_switches2_y.shape[3]
	assert output_switches3_x.shape[2] == output_switches3_y.shape[3]
	
	assert output_switches1_x_s31.shape[0] == output_switches2_x_s31.shape[0] == output_switches3_x_s31.shape[0] == FL.shape[0]
	assert output_switches1_x_s31.shape[2] == output_switches1_y_s31.shape[3]
	assert output_switches2_x_s31.shape[2] == output_switches2_y_s31.shape[3]
	assert output_switches3_x_s31.shape[2] == output_switches3_y_s31.shape[3]
	
	assert output_switches1_x_s31.shape[1] == output_switches1_x.shape[1]
	assert output_switches1_x_s31.shape[2] == output_switches1_x.shape[2]
	assert output_switches1_x_s31.shape[3] == output_switches1_x.shape[3]
	
	assert output_switches2_x_s31.shape[1] == output_switches2_x.shape[1]
	assert output_switches2_x_s31.shape[2] == output_switches2_x.shape[2]
	assert output_switches2_x_s31.shape[3] == output_switches2_x.shape[3]
	
	assert output_switches3_x_s31.shape[1] == output_switches3_x.shape[1]
	assert output_switches3_x_s31.shape[2] == output_switches3_x.shape[2]
	assert output_switches3_x_s31.shape[3] == output_switches3_x.shape[3]
	
	assert imgs.shape[2] == imgs.shape[3]
	
	N_IMGS = imgs.shape[0]
	n3 = output_switches3_x.shape[1]
	n2 = output_switches2_x.shape[1]
	n1 = output_switches1_x.shape[1]
	max_output_sz3 = output_switches3_x.shape[2]
	
	if deriv_ind == 1:
		F1 = np.ones_like(F1)
	elif deriv_ind == 2:
		F2 = np.ones_like(F2)
	elif deriv_ind == 3:
		F3 = np.ones_like(F3)
	elif deriv_ind == 4:
		FL = np.ones_like(FL)
	
	if not F1.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F1)'
		F1 = np.ascontiguousarray(F1)
	
	if not F2.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F2)'
		F2 = np.ascontiguousarray(F2)
	
	if not F3.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F3)'
		F3 = np.ascontiguousarray(F3)
	
	if not FL.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (FL)'
		FL = np.ascontiguousarray(FL)
	
	
	if not imgs.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	
	############################
	
	if not output_switches3_x_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x_s31)'
		output_switches3_x_s31 = np.ascontiguousarray(output_switches3_x_s31)
	if not output_switches3_y_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y_s31)'
		output_switches3_y_s31 = np.ascontiguousarray(output_switches3_y_s31)
		
	if not output_switches2_x_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x_s31)'
		output_switches2_x_s31 = np.ascontiguousarray(output_switches2_x_s31)
	if not output_switches2_y_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y_s31)'
		output_switches2_y_s31 = np.ascontiguousarray(output_switches2_y_s31)
		
	if not output_switches1_x_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x_s31)'
		output_switches1_x_s31 = np.ascontiguousarray(output_switches1_x_s31)
	if not output_switches1_y_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y_s31)'
		output_switches1_y_s31 = np.ascontiguousarray(output_switches1_y_s31)
	
	return _sigma31_layers.bp_patch_sigma31(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x_s31, output_switches3_y_s31, output_switches2_x_s31, output_switches2_y_s31, output_switches1_x_s31, output_switches1_y_s31, imgs, sigma_imgs, deriv_ind, pred, F1, F2, F3, FL)


def bp_patch_sigma31_gpu(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x_s31, output_switches3_y_s31, output_switches2_x_s31, output_switches2_y_s31, output_switches1_x_s31, output_switches1_y_s31, imgs, sigma_imgs, deriv_ind, pred, F1, F2, F3, FL, gpu_ind=0, warn=True):
	assert output_switches3_x.dtype == np.dtype('int64')
	assert output_switches3_y.dtype == np.dtype('int64')
	assert output_switches2_x.dtype == np.dtype('int64')
	assert output_switches2_y.dtype == np.dtype('int64')
	assert output_switches1_x.dtype == np.dtype('int64')
	assert output_switches1_y.dtype == np.dtype('int64')
	
	assert output_switches3_x_s31.dtype == np.dtype('int64')
	assert output_switches3_y_s31.dtype == np.dtype('int64')
	assert output_switches2_x_s31.dtype == np.dtype('int64')
	assert output_switches2_y_s31.dtype == np.dtype('int64')
	assert output_switches1_x_s31.dtype == np.dtype('int64')
	assert output_switches1_y_s31.dtype == np.dtype('int64')
	
	assert isinstance(gpu_ind,int)
	assert isinstance(deriv_ind,int)
	
	assert F1.dtype == np.dtype('float32')
	assert F2.dtype == np.dtype('float32')
	assert F3.dtype == np.dtype('float32')
	assert FL.dtype == np.dtype('float32')

	assert F1.shape[-1] == F1.shape[-2]
	assert F2.shape[-1] == F2.shape[-2]
	assert F3.shape[-1] == F3.shape[-2]
	assert FL.shape[-1] == FL.shape[-2]
	
	assert F2.shape[1] == F1.shape[0]
	assert F2.shape[1] == F3.shape[0]
	assert FL.shape[1] == F3.shape[0]
	
	assert output_switches1_x.shape[0] == output_switches2_x.shape[0] == output_switches3_x.shape[0] == imgs.shape[0]
	assert output_switches1_x.shape[2] == output_switches1_y.shape[3]
	assert output_switches2_x.shape[2] == output_switches2_y.shape[3]
	assert output_switches3_x.shape[2] == output_switches3_y.shape[3]
	
	assert output_switches1_x_s31.shape[0] == output_switches2_x_s31.shape[0] == output_switches3_x_s31.shape[0] == FL.shape[0]
	assert output_switches1_x_s31.shape[2] == output_switches1_y_s31.shape[3]
	assert output_switches2_x_s31.shape[2] == output_switches2_y_s31.shape[3]
	assert output_switches3_x_s31.shape[2] == output_switches3_y_s31.shape[3]
	
	assert output_switches1_x_s31.shape[1] == output_switches1_x.shape[1]
	assert output_switches1_x_s31.shape[2] == output_switches1_x.shape[2]
	assert output_switches1_x_s31.shape[3] == output_switches1_x.shape[3]
	
	assert output_switches2_x_s31.shape[1] == output_switches2_x.shape[1]
	assert output_switches2_x_s31.shape[2] == output_switches2_x.shape[2]
	assert output_switches2_x_s31.shape[3] == output_switches2_x.shape[3]
	
	assert output_switches3_x_s31.shape[1] == output_switches3_x.shape[1]
	assert output_switches3_x_s31.shape[2] == output_switches3_x.shape[2]
	assert output_switches3_x_s31.shape[3] == output_switches3_x.shape[3]
	
	assert imgs.shape[2] == imgs.shape[3]
	
	N_IMGS = imgs.shape[0]
	n3 = output_switches3_x.shape[1]
	n2 = output_switches2_x.shape[1]
	n1 = output_switches1_x.shape[1]
	max_output_sz3 = output_switches3_x.shape[2]
	
	if deriv_ind == 1:
		F1 = np.ones_like(F1)
	elif deriv_ind == 2:
		F2 = np.ones_like(F2)
	elif deriv_ind == 3:
		F3 = np.ones_like(F3)
	elif deriv_ind == 4:
		FL = np.ones_like(FL)
	
	if not F1.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F1)'
		F1 = np.ascontiguousarray(F1)
	
	if not F2.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F2)'
		F2 = np.ascontiguousarray(F2)
	
	if not F3.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F3)'
		F3 = np.ascontiguousarray(F3)
	
	if not FL.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (FL)'
		FL = np.ascontiguousarray(FL)
	
	
	if not imgs.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	
	############################
	
	if not output_switches3_x_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x_s31)'
		output_switches3_x_s31 = np.ascontiguousarray(output_switches3_x_s31)
	if not output_switches3_y_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y_s31)'
		output_switches3_y_s31 = np.ascontiguousarray(output_switches3_y_s31)
		
	if not output_switches2_x_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x_s31)'
		output_switches2_x_s31 = np.ascontiguousarray(output_switches2_x_s31)
	if not output_switches2_y_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y_s31)'
		output_switches2_y_s31 = np.ascontiguousarray(output_switches2_y_s31)
		
	if not output_switches1_x_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x_s31)'
		output_switches1_x_s31 = np.ascontiguousarray(output_switches1_x_s31)
	if not output_switches1_y_s31.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y_s31)'
		output_switches1_y_s31 = np.ascontiguousarray(output_switches1_y_s31)
	
	return _sigma31_layers.bp_patch_sigma31_gpu(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, output_switches3_x_s31, output_switches3_y_s31, output_switches2_x_s31, output_switches2_y_s31, output_switches1_x_s31, output_switches1_y_s31, imgs, sigma_imgs, deriv_ind, pred, F1, F2, F3, FL, gpu_ind)

# output_switches3_x, output_switches3_y, [n_imgs, n3, max_output_sz3, max_output_sz3]
# output_switches2_x, output_switches2_y, [n_imgs, n2, max_output_sz2, max_output_sz2]
# output_switches1_x, output_switches1_y, [n_imgs, n1, max_output_sz1, max_output_sz1]
# ints: s1, s2, s3
# labels [n_imgs]
# imgs: [n_imgs, 3, img_sz, img_sz] (float32)
# int: N_C
def patch_inds(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs, N_C, inds, warn=True):
	assert output_switches3_x.dtype == np.dtype('int64')
	assert output_switches3_y.dtype == np.dtype('int64')
	assert output_switches2_x.dtype == np.dtype('int64')
	assert output_switches2_y.dtype == np.dtype('int64')
	assert output_switches1_x.dtype == np.dtype('int64')
	assert output_switches1_y.dtype == np.dtype('int64')
	assert inds.dtype == np.dtype('int64')
	assert np.min(inds) >= 0
	assert len(inds.shape) == 1
	
	
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
	
	assert np.max(inds) < (n1*3*s1*s1*n2*s2*s2*n3*s3*s3*max_output_sz3*max_output_sz3)
	
	if not imgs.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	
	if not inds.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (inds)'
		inds = np.ascontiguousarray(inds)
	
	return _sigma31_layers.compute_patch_inds(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs, N_C, inds)

def pred_deriv_gpu(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, imgs, N_C, deriv_layer_ind, pred, warn=True):
	assert output_switches3_x.dtype == np.dtype('int64')
	assert output_switches3_y.dtype == np.dtype('int64')
	assert output_switches2_x.dtype == np.dtype('int64')
	assert output_switches2_y.dtype == np.dtype('int64')
	assert output_switches1_x.dtype == np.dtype('int64')
	assert output_switches1_y.dtype == np.dtype('int64')
	assert isinstance(deriv_layer_ind,int)
	assert F1.dtype == np.dtype('float32')
	assert F2.dtype == np.dtype('float32')
	assert F3.dtype == np.dtype('float32')
	assert FL.dtype == np.dtype('float32')
	assert pred.dtype == np.dtype('float32')
	
	assert pred.shape[0] == FL.shape[0]
	assert pred.shape[1] == imgs.shape[0]
	
	assert F1.shape[-1] == F1.shape[-2]
	assert F2.shape[-1] == F2.shape[-2]
	assert F3.shape[-1] == F3.shape[-2]
	assert FL.shape[-1] == FL.shape[-2]
	
	assert F2.shape[1] == F1.shape[0]
	assert F2.shape[1] == F3.shape[0]
	assert FL.shape[1] == F3.shape[0]
	
	assert isinstance(s1,int)
	assert isinstance(s2,int)
	assert isinstance(s3,int)
	assert isinstance(N_C,int)
	assert output_switches1_x.shape[0] == output_switches2_x.shape[0] == output_switches3_x.shape[0] == imgs.shape[0]
	assert output_switches1_x.shape[2] == output_switches1_y.shape[3]
	assert output_switches2_x.shape[2] == output_switches2_y.shape[3]
	assert output_switches3_x.shape[2] == output_switches3_y.shape[3]
	assert imgs.shape[2] == imgs.shape[3]
	
	N_IMGS = imgs.shape[0]
	n3 = output_switches3_x.shape[1]
	n2 = output_switches2_x.shape[1]
	n1 = output_switches1_x.shape[1]
	max_output_sz3 = output_switches3_x.shape[2]
	
	if not F1.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F1)'
		F1 = np.ascontiguousarray(F1)
	
	if not F2.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F2)'
		F2 = np.ascontiguousarray(F2)
	
	if not F3.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (F3)'
		F3 = np.ascontiguousarray(F3)
	
	if not FL.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (FL)'
		FL = np.ascontiguousarray(FL)
	if not imgs.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	if not pred.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (pred)'
		pred = np.ascontiguousarray(pred)
	
	return _sigma31_layers.pred_deriv_gpu(F1, F2, F3, FL, output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, imgs, N_C, deriv_layer_ind, pred)

# output_switches3_x, output_switches3_y, [n_imgs, n3, max_output_sz3, max_output_sz3]
# output_switches2_x, output_switches2_y, [n_imgs, n2, max_output_sz2, max_output_sz2]
# output_switches1_x, output_switches1_y, [n_imgs, n1, max_output_sz1, max_output_sz1]
# ints: s1, s2, s3
# labels [n_imgs]
# imgs: [n_imgs, 3, img_sz, img_sz] (float32)
# int: N_C
def s31_full_gpu(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs, N_C, warn=True):
	assert output_switches3_x.dtype == np.dtype('int64')
	assert output_switches3_y.dtype == np.dtype('int64')
	assert output_switches2_x.dtype == np.dtype('int64')
	assert output_switches2_y.dtype == np.dtype('int64')
	assert output_switches1_x.dtype == np.dtype('int64')
	assert output_switches1_y.dtype == np.dtype('int64')
	
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
		if warn:
			print 'warning: input not C-contiguous (imgs)'
		imgs = np.ascontiguousarray(imgs)
	
	if not output_switches3_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_x)'
		output_switches3_x = np.ascontiguousarray(output_switches3_x)
	if not output_switches3_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches3_y)'
		output_switches3_y = np.ascontiguousarray(output_switches3_y)
		
	if not output_switches2_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_x)'
		output_switches2_x = np.ascontiguousarray(output_switches2_x)
	if not output_switches2_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches2_y)'
		output_switches2_y = np.ascontiguousarray(output_switches2_y)
		
	if not output_switches1_x.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_x)'
		output_switches1_x = np.ascontiguousarray(output_switches1_x)
	if not output_switches1_y.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (output_switches1_y)'
		output_switches1_y = np.ascontiguousarray(output_switches1_y)
	
	return _sigma31_layers.compute_sigma31_full_gpu(output_switches3_x, output_switches3_y, output_switches2_x, output_switches2_y, output_switches1_x, output_switches1_y, s1, s2, s3, labels, imgs, N_C)

def einsum_deriv_gpu(deriv_layer_ind, sigma_ind, output_ind, gpu_ind):
	assert isinstance(deriv_layer_ind,int)
	assert isinstance(gpu_ind,int)
	assert isinstance(output_ind,int)
	assert isinstance(sigma_ind,int)
	
	return _sigma31_layers.einsum_deriv_gpu(deriv_layer_ind, sigma_ind, output_ind, gpu_ind)

def einsum_return(output_ind, gpu_ind):
	assert isinstance(gpu_ind, int)
	assert isinstance(output_ind, int)
	
	return _sigma31_layers.einsum_return(output_ind, gpu_ind)

def set_sigma_buffer(sigma31, layer_ind, gpu_ind):
	# sigma: N_C, n1, n0, s1, s1, n2, s2, s2, n3, s3, s3, max_output_sz3, max_output_sz3
	
	# spatial dims should be the same (both broadcasted or not)
	assert sigma31.shape[3] == sigma31.shape[4]
	assert sigma31.shape[6] == sigma31.shape[7]
	assert sigma31.shape[9] == sigma31.shape[10]
	assert sigma31.shape[-1] == sigma31.shape[-2]
	
	assert (sigma31.shape[1] == sigma31.shape[5] or sigma31.shape[1] == 1 or sigma31.shape[5] == 1) and (sigma31.shape[5] == sigma31.shape[-5] or sigma31.shape[5] == 1 or sigma31.shape[-5] == 1) and (sigma31.shape[1] == sigma31.shape[-5] or sigma31.shape[1] == 1 or sigma31.shape[-5] == 1)
	
	assert isinstance(layer_ind,int)
	assert isinstance(gpu_ind,int)
	
	assert sigma31.dtype == np.dtype('float32')
	
	if not sigma31.flags.contiguous:
		print 'warning: input not C-contiguous (sigma31)'
		sigma31 = np.ascontiguousarray(sigma31)
	
	return _sigma31_layers.set_sigma_buffer(sigma31, layer_ind, gpu_ind, 0) # last arg is for showing warnings of previously set sigma buffers

def set_sigma11_buffer(sigma11, inds, gpu_ind):
	assert len(sigma11.shape) == 1
	assert len(inds.shape) == 1
	assert inds.dtype == np.dtype('int64')
	assert np.min(inds) >= 0
	
	assert isinstance(gpu_ind,int)
	
	assert sigma11.dtype == np.dtype('float32')
	
	if not sigma11.flags.contiguous:
		print 'warning: input not C-contiguous (sigma11)'
		sigma11 = np.ascontiguousarray(sigma11)
		
	if not inds.flags.contiguous:
		print 'warning: input not C-contiguous (inds)'
		inds = np.ascontiguousarray(inds)
	
	return _sigma31_layers.set_sigma11_buffer(sigma11, inds, gpu_ind)

def F_layer_sum_deriv_inds_gpu_return(layer_ind, gpu_ind):
	assert isinstance(layer_ind,int)
	assert isinstance(gpu_ind,int)
	
	return _sigma31_layers.compute_F_layer_sum_deriv_inds_gpu_return(layer_ind, gpu_ind)

def set_FL321_buffer(FL321, gpu_ind):
	assert len(FL321.shape) == 2
	
	assert isinstance(gpu_ind,int)
	
	assert FL321.dtype == np.dtype('float32')
	
	if not FL321.flags.contiguous:
		print 'warning: input not C-contiguous (FL321)'
		FL321 = np.ascontiguousarray(FL321)
	
	return _sigma31_layers.set_FL321_buffer(FL321, gpu_ind)

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

def max_pool_locs(conv_output, PAD=0, warn=True):
	assert conv_output.shape[2] == conv_output.shape[3]
	assert PAD >= 0
	assert isinstance(PAD,int)
	assert conv_output.dtype == np.dtype('float32')
	
	if not conv_output.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (conv_output)'
		conv_output = np.ascontiguousarray(conv_output)
	
	return _sigma31_layers.max_pool_locs(conv_output, PAD)

def compute_sigma11(patches, warn=True):
	assert len(patches.shape) == 2
	assert patches.dtype == np.dtype('float32')
	
	if not patches.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (patches)'
		patches = np.ascontiguousarray(patches)
	
	return _sigma31_layers.compute_sigma11(patches)

def compute_sigma11_gpu(patches, warn=True):
	assert len(patches.shape) == 2
	assert patches.dtype == np.dtype('float32')
	
	if not patches.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (patches)'
		patches = np.ascontiguousarray(patches)
	
	return _sigma31_layers.compute_sigma11_gpu(patches)

def compute_sigma11_lin_gpu(patches, warn=True):
	assert len(patches.shape) == 2
	assert patches.dtype == np.dtype('float32')
	
	if not patches.flags.contiguous:
		if warn:
			print 'warning: input not C-contiguous (patches)'
		patches = np.ascontiguousarray(patches)
	
	return _sigma31_layers.compute_sigma11_lin_gpu(patches)