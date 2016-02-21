import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0,0]

# additional_args= [PAD]
def conv(args, OUT_BUFFER=None, additional_args=[0], gpu_ind=GPU_IND):
	t = time.time()
	
	F, IMGS = args
	PAD = additional_args[0]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	OUT_BUFFER[1] = _ntm_module3.conv(F[0], F[1], IMGS[0], IMGS[1], PAD, OUT_BUFFER[0], gpu_ind)
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

def conv_ddata(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[0], gpu_ind=GPU_IND):
	t = time.time()
	
	F, IMGS = args
	
	PAD = additional_args[0]
	
	# collapse dims that should not be summed over
	# ex. DERIV_ABOVE = (a,b,c,d,e,f)
	# ex. LAYER_OUT = (d,e,f)
	# -> DERIV_ABOVE = (a*b*c, d,e,f)
	n_dims_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1]) + 1 # plus 1 (the image dimension)
	DERIV_ABOVE_reshaped = (np.prod(DERIV_ABOVE[1][:n_dims_not_summed]),) + LAYER_OUT[1][1:]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	OUT_BUFFER[1] = _ntm_module3.conv_ddata(F[0], F[1], IMGS[0], IMGS[1], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, PAD, OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dims_not_summed] + IMGS[1][1:]
	
	t_main[1] += time.time() - t
	return OUT_BUFFER
	
def conv_dfilter(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[0], gpu_ind=GPU_IND):
	t = time.time()
	
	F, IMGS = args
	
	PAD = additional_args[0]
	
	# collapse dims that should not be summed over
	# ex. DERIV_ABOVE = (a,b,c,d,e,f)
	# ex. LAYER_OUT = (d,e,f)
	# -> DERIV_ABOVE = (a*b*c, d,e,f)
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	n_imgs = DERIV_ABOVE[1][0]
	dim_above = np.prod(DERIV_ABOVE[1][1:+n_dim_not_summed])
	DERIV_ABOVE_reshaped = (n_imgs, dim_above) + LAYER_OUT[1][1:]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	OUT_BUFFER[1] = _ntm_module3.conv_dfilter(F[0], F[1], IMGS[0], IMGS[1], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, PAD, OUT_BUFFER[0], gpu_ind)
	
	# did we sum all the images together or not?
	OUT_BUFFER[1] = F[1]
	if dim_above != 1:
		OUT_BUFFER[1] = DERIV_ABOVE[1][:n_dim_not_summed+1] + F[1]
	
	t_main[2] += time.time() - t
	return OUT_BUFFER
	

# source = None: source is previous layer
# source = -1: source is user-supplied
# source = str: source is another layer
def add_conv_layer(LAYERS, name, n_filters, filter_sz, source=None, imgs_shape=None, random_function=random_normal_function, PAD=0, init=0):
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
		in_prev = [False, False]
		
		source_meta[0] = random_function
		
		if source is None: # previous layer
			source_meta[1] = layer_ind-1
			in_shape[1] = LAYERS[layer_ind-1]['out_shape']
		elif source == -1: # user supplied
			source_meta[1] = -1
			in_shape[1] = imgs_shape
			assert len(imgs_shape) == 4
			#assert imgs_shape[0] == 1
		elif isinstance(source,str):
			source_meta[1] = find_layer(LAYERS, source)
			assert source_meta[1] is not None, 'could not find source conv inputs'
			
			if source[-1] == '-':
				in_shape[1] = imgs_shape
				assert len(imgs_shape) == 4
				assert imgs_shape[0] == 1
				in_prev[1] = True
			else:	
				in_shape[1] = LAYERS[source_meta[1]]['out_shape']

		n_channels = in_shape[1][1]
		in_shape[0] = (n_filters, n_channels, filter_sz, filter_sz)
		
		# empirically determine output shape
		F_temp = init_buffer(np.zeros(in_shape[0], dtype='single'))
		IMGS_temp = init_buffer(np.zeros(in_shape[1], dtype='single'))
		
		O = conv((F_temp, IMGS_temp), additional_args=[PAD])
		out_shape = copy.deepcopy(O[1])
		
		free_buffer(O)
		free_buffer(F_temp)
		free_buffer(IMGS_temp)
		
		LAYERS[layer_ind]['forward_F'] = conv
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = in_shape
		LAYERS[layer_ind]['in_source'] = source_meta
		LAYERS[layer_ind]['deriv_F'] = [conv_dfilter, conv_ddata]
		LAYERS[layer_ind]['in_prev'] = in_prev
		LAYERS[layer_ind]['additional_forward_args'] = [PAD]
		LAYERS[layer_ind]['additional_deriv_args'] = [[PAD], [PAD]]
		
		return layer_ind
