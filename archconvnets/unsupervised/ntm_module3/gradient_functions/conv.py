import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *

# additional_args= [PAD]
def conv(args, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, IMGS = args
	check_buffer(F)
	check_buffer(IMGS)
	n_filters, n_channels, filter_sz, filter_sz2  = F[1]
	n_imgs, n_channels2, img_sz, img_sz2 = IMGS[1]
	PAD = additional_args[0]
	assert isinstance(PAD,int)
	assert PAD >= 0
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert IMGS[1][0] == 1
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		OUT_BUFFER[1] = _ntm_module3.conv(F[0], F[1], IMGS[0], IMGS[1], PAD, OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		assert False, 'cpu conv not supported'
		
	return OUT_BUFFER

def conv_ddata(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, IMGS = args
	check_buffer(F)
	check_buffer(IMGS)
	check_buffer(DERIV_ABOVE)
	n_filters, n_channels, filter_sz, filter_sz2  = F[1]
	n_imgs, n_channels2, img_sz, img_sz2 = IMGS[1]
	PAD = additional_args[0]
	assert isinstance(PAD,int)
	assert PAD >= 0
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert IMGS[1][0] == LAYER_OUT[1][0]
	assert F[1][0] == LAYER_OUT[1][1]
	assert LAYER_OUT[1][0] == IMGS[1][0] == 1
	
	# collapse dims that should not be summed over
	# ex. DERIV_ABOVE = (a,b,c,d,e,f)
	# ex. LAYER_OUT = (d,e,f)
	# -> DERIV_ABOVE = (a*b*c, d,e,f)
	DERIV_ABOVE_reshaped = copy.deepcopy(DERIV_ABOVE)
	n_dims_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	DERIV_ABOVE_reshaped[1] = tuple(np.concatenate((np.prod(DERIV_ABOVE_reshaped[1][:n_dims_not_summed])[np.newaxis], LAYER_OUT[1][1:])))
	assert DERIV_ABOVE[1][n_dims_not_summed:] == LAYER_OUT[1]
	check_buffer(DERIV_ABOVE_reshaped)
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		OUT_BUFFER[1] = _ntm_module3.conv_ddata(F[0], F[1], IMGS[0], IMGS[1], DERIV_ABOVE[0], DERIV_ABOVE_reshaped[1], PAD, OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		assert False, 'cpu conv not supported'
	
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dims_not_summed], IMGS[1])))
	check_buffer(OUT_BUFFER)
	
	return OUT_BUFFER
	
def conv_dfilter(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[0], gpu_ind=0):
	assert isinstance(gpu_ind,int)
	F, IMGS = args
	check_buffer(F)
	check_buffer(IMGS)
	check_buffer(DERIV_ABOVE)
	n_filters, n_channels, filter_sz, filter_sz2  = F[1]
	n_imgs, n_channels2, img_sz, img_sz2 = IMGS[1]
	PAD = additional_args[0]
	assert isinstance(PAD,int)
	assert PAD >= 0
	assert n_channels == n_channels2 and img_sz == img_sz2 and filter_sz == filter_sz2
	assert IMGS[1][0] == LAYER_OUT[1][0]
	assert F[1][0] == LAYER_OUT[1][1]
	assert LAYER_OUT[1][0] == IMGS[1][0] == 1
	
	# collapse dims that should not be summed over
	# ex. DERIV_ABOVE = (a,b,c,d,e,f)
	# ex. LAYER_OUT = (d,e,f)
	# -> DERIV_ABOVE = (a*b*c, d,e,f)
	DERIV_ABOVE_reshaped = copy.deepcopy(DERIV_ABOVE)
	n_dims_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	DERIV_ABOVE_reshaped[1] = tuple(np.concatenate((np.prod(DERIV_ABOVE_reshaped[1][:n_dims_not_summed])[np.newaxis], LAYER_OUT[1][1:])))
	assert DERIV_ABOVE[1][n_dims_not_summed:] == LAYER_OUT[1]
	check_buffer(DERIV_ABOVE_reshaped)
	
	if OUT_BUFFER != None:
		check_buffer(OUT_BUFFER)
	else:
		OUT_BUFFER = init_buffer()
	
	if GPU:
		OUT_BUFFER[1] = _ntm_module3.conv_dfilter(F[0], F[1], IMGS[0], IMGS[1], DERIV_ABOVE[0], DERIV_ABOVE_reshaped[1], PAD, OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		assert False, 'cpu conv not supported'
	
	OUT_BUFFER[1] = tuple(np.concatenate((DERIV_ABOVE[1][:n_dims_not_summed], F[1])))
	
	check_buffer(OUT_BUFFER)
	
	return OUT_BUFFER
	

# source = None: source is previous layer
# source = -1: source is user-supplied
# source = str: source is another layer
def add_conv_layer(LAYERS, name, n_filters, filter_sz, source=None, imgs_shape=None, random_function=random_function, PAD=0, init=0):
	assert isinstance(n_filters, int)
	assert isinstance(filter_sz, int)
	assert isinstance(name, str)
	assert isinstance(PAD, int)
	assert PAD >= 0
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		in_shape = [None]*2
		source_meta = [None]*2
		
		source_meta[0] = random_function
		
		if source is None: # previous layer
			source_meta[1] = layer_ind-1
			in_shape[1] = LAYERS[layer_ind-1]['out_shape']
		elif source == -1: # user supplied
			source_meta[1] = -1
			in_shape[1] = imgs_shape
			assert len(imgs_shape) == 4
			assert imgs_shape[0] == 1
		elif isinstance(source,str):
			source[1] = find_layer(LAYERS, source[1])
			assert source[1] is not None, 'could not find source conv inputs'
			in_shape[1] = LAYERS[source[1]]['out_shape']

		n_channels = in_shape[1][1]
		in_shape[0] = (n_filters, n_channels, filter_sz, filter_sz)
		
		# empirically determine output shape
		F_temp = init_buffer(np.zeros(in_shape[0], dtype='single'))
		IMGS_temp = init_buffer(np.zeros(in_shape[1], dtype='single'))
		
		O = conv((F_temp, IMGS_temp))
		out_shape = copy.deepcopy(O[1])
		
		free_buffer(O)
		free_buffer(F_temp)
		free_buffer(IMGS_temp)
		
		LAYERS[layer_ind]['forward_F'] = conv
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = source_meta
		LAYERS[layer_ind]['deriv_F'] = [conv_dfilter, conv_ddata]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		LAYERS[layer_ind]['additional_forward_args'] = [PAD]
		LAYERS[layer_ind]['additional_deriv_args'] = [[PAD], [PAD]]
		
		return layer_ind
