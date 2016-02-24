import numpy as np
import time
import scipy.optimize
from ntm_core import *
#from architectures.model_architecture_movie_no_mem import init_model
#from architectures.model_architecture_movie import init_model
#from architectures.model_architecture_simple import init_model
#from architectures.model_architecture_cp import init_model
#from architectures.movie_phys_latent import init_model
#from architectures.model_architecture_cp_batched import init_model
#from architectures.model_architecture_movie_mem_batched import init_model
from architectures.model_architecture_movie_lstm_conv_batched import init_model


free_all_buffers()
N_FRAMES = 10 #50*2

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()[:4]

F1_IND = 0
STACK_SUM_PX_IND = find_layer(LAYERS, 'STACK_SUM_PX_lin')
TARGET_IND = find_layer(LAYERS, 'ERR')

x1t = random_function(np.concatenate(((N_FRAMES,), LAYERS[F1_IND]['in_shape'][1]))) / 10
target = random_function(np.concatenate(((N_FRAMES,), LAYERS[TARGET_IND]['in_shape'][1]))) / 10
set_buffer(target[0], WEIGHTS[TARGET_IND][1]) # target

OUTPUT = None; WEIGHT_DERIVS = None
MEM_DERIVS = [None]*len(MEM_INDS)
OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)

PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

t_start = time.time()

for frame in range(N_FRAMES):
	set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs
	set_buffer(x1t[frame], WEIGHTS[STACK_SUM_PX_IND][1])  # inputs
	set_buffer(target[frame], WEIGHTS[TARGET_IND][1])  # inputs
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
	
	# update partials_prev
	MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
	PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
	
	OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
print 'elapsed time', time.time() - t_start

import archconvnets.unsupervised.ntm_module3.gradient_functions.cosine_sim as cosine_sim_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.linear_F as linear_F_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.add_points as add_points_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.sum_points as sum_points_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.focus_key as focus_key_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.sigmoid as sigmoid_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.sharpen as sharpen_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.relu as relu_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.shift_w as shift_w_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.interpolate as interpolate_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.softmax as softmax_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.sq_points as sq_points_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.dotT as dotT_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.mult_points as mult_points_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.conv as conv_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.max_pool as max_pool_module
import archconvnets.unsupervised.ntm_module3.gradient_functions.pearson as pearson_module

print
print 'point_wise_add', t_add
print 'dot', t_dot
print 'mult_partials_keep', t_mult_partials_keep
print 'mult_partials_nkeep', t_mult_partials_nkeep
print 'cosine_sim', cosine_sim_module.t_main
print 'linear_F', linear_F_module.t_main
print 'add_points', add_points_module.t_main
print 'sum_points', sum_points_module.t_main
print 'focus_key', focus_key_module.t_main
print 'sigmoid', sigmoid_module.t_main
print 'sharpen', sharpen_module.t_main
print 'relu', relu_module.t_main
print 'shift_w', shift_w_module.t_main
print 'interpolate', interpolate_module.t_main
print 'softmax', softmax_module.t_main
print 'sq_points', sq_points_module.t_main
print 'dotT', dotT_module.t_main
print 'mult_points', mult_points_module.t_main
print 'conv', conv_module.t_main
print 'max_pool', max_pool_module.t_main
print 'pearson', pearson_module.t_main
