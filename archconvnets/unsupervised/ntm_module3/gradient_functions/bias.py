import numpy as np
import archconvnets.unsupervised.ntm_module3._ntm_module3 as _ntm_module3
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *
from archconvnets.unsupervised.ntm3.gpu_flag import *
from archconvnets.unsupervised.ntm3.ntm_core import *

# c = a + scalar*b
def add_bias_layer(LAYERS, name, batch_imgs=False, random_function=random_function, init=0):
	assert isinstance(name, str)
	
	if init == 0:
		assert find_layer(LAYERS, name) is None, 'layer %s has already been added' % name
		LAYERS.append({'name': name})
		return len(LAYERS)-1
	else:
		layer_ind = find_layer(LAYERS, name)
		assert layer_ind is not None, 'layer %s has not already been added' % name
		
		out_shape = LAYERS[layer_ind-1]['out_shape']
		
		LAYERS[layer_ind]['out_shape'] = out_shape
		LAYERS[layer_ind]['in_source'] = [layer_ind-1, random_function]
		LAYERS[layer_ind]['in_prev'] = [False, False]
		
		if batch_imgs:
			LAYERS[layer_ind]['forward_F'] = add_points_batch
			LAYERS[layer_ind]['in_shape'] = [out_shape, out_shape[1:]]
			LAYERS[layer_ind]['deriv_F'] = [add_points_dinput, add_points_batch_dinputB]
			LAYERS[layer_ind]['additional_forward_args'] = [None]
			LAYERS[layer_ind]['additional_deriv_args'] = [[1], [None]]
			
		else:
			LAYERS[layer_ind]['forward_F'] = add_points
			LAYERS[layer_ind]['in_shape'] = [out_shape, out_shape]
			LAYERS[layer_ind]['deriv_F'] = [add_points_dinput, add_points_dinput]
			LAYERS[layer_ind]['additional_forward_args'] = [1]
			LAYERS[layer_ind]['additional_deriv_args'] = [[1], [1]]
		
		return layer_ind

def add_linear_F_bias_layer(LAYERS, name, n_filters, source=None, sum_all=False, squeeze=False, batch_imgs=False, random_function=random_normal_function, init=0):
	add_linear_F_layer(LAYERS, name+'_lin', n_filters, source, sum_all, squeeze, batch_imgs, random_function, init)
	add_bias_layer(LAYERS, name, batch_imgs, random_function, init)
	
def add_sigmoid_F_bias_layer(LAYERS, name, n_filters, source=None, sum_all=False, squeeze=False, batch_imgs=False, random_function=random_function, init=0):
	if isinstance(n_filters,int) == False:
		squeeze = True
	add_linear_F_layer(LAYERS, name+'_lin', n_filters, source, sum_all, squeeze, batch_imgs, random_function, init)
	add_bias_layer(LAYERS, name+'_b', batch_imgs, random_function, init)
	add_sigmoid_layer(LAYERS, name, init=init)
	
def add_relu_F_bias_layer(LAYERS, name, n_filters, source=None, sum_all=False, squeeze=False, batch_imgs=False, random_function=random_function, init=0):
	add_linear_F_layer(LAYERS, name+'_lin', n_filters, source, sum_all, squeeze, batch_imgs, random_function, init)
	add_bias_layer(LAYERS, name+'_b', batch_imgs, random_function, init)
	add_relu_layer(LAYERS, name, init=init)
	