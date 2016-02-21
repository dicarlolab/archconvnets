import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *
import time

t_main = [0,0]

def max_pool(args, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	CONV_OUTPUT = args[0]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	OUT_BUFFER[1] = _ntm_module3.max_pool(CONV_OUTPUT[0], CONV_OUTPUT[1], OUT_BUFFER[0], gpu_ind)
	
	t_main[0] += time.time() - t
	return OUT_BUFFER

# srcData = LAYER_OUT
# destData = args[0]
def max_pool_dinput(args, LAYER_OUT, DERIV_ABOVE, OUT_BUFFER=None, additional_args=[None], gpu_ind=GPU_IND):
	t = time.time()
	
	DESTDATA = args[0]
	SRCDATA = LAYER_OUT
	
	# collapse dims that should not be summed over
	# ex. DERIV_ABOVE = (a,b,c,d,e,f)
	# ex. LAYER_OUT = (d,e,f)
	# -> DERIV_ABOVE = (a*b*c, d,e,f)
	n_dim_not_summed = len(DERIV_ABOVE[1]) - len(LAYER_OUT[1])
	n_imgs = DERIV_ABOVE[1][0]
	dim_above = np.prod(DERIV_ABOVE[1][1:1+n_dim_not_summed])
	DERIV_ABOVE_reshaped = (n_imgs, dim_above) + LAYER_OUT[1][1:]
	
	if OUT_BUFFER is None:
		OUT_BUFFER = init_buffer()
	
	OUT_BUFFER[1] = _ntm_module3.max_pool_dinput(DESTDATA[0], DESTDATA[1], SRCDATA[0], DERIV_ABOVE[0], DERIV_ABOVE_reshaped, OUT_BUFFER[0], gpu_ind)
	
	OUT_BUFFER[1] = DERIV_ABOVE[1][:1+n_dim_not_summed] + DESTDATA[1][1:]

	t_main[1] += time.time() - t
	return OUT_BUFFER


# source = None: source is previous layer
# source = -1: source is user-supplied
# source = str: source is another layer
def add_max_pool_layer(LAYERS, name, init=0):
	assert isinstance(name, str)
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		assert layer_ind >= 0
		
		in_shape = LAYERS[layer_ind-1]['out_shape']
		
		# empirically determine output shape
		IMGS_temp = init_buffer(np.zeros(in_shape, dtype='single'))
		
		O = max_pool((IMGS_temp,))
		out_shape = copy.deepcopy(O[1])
		
		free_buffer(O)
		free_buffer(IMGS_temp)
		
		LAYERS[layer_ind]['forward_F'] = max_pool
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_shape'] = [in_shape]
		LAYERS[layer_ind]['in_source'] = [layer_ind-1]
		LAYERS[layer_ind]['deriv_F'] = [max_pool_dinput]
		LAYERS[layer_ind]['in_prev'] = [False]
		LAYERS[layer_ind]['additional_forward_args'] = [None]
		LAYERS[layer_ind]['additional_deriv_args'] = [[None]]
		
		return layer_ind
