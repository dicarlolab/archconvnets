import numpy as np
import time
import scipy.optimize
from ntm_core import *
#from model_architecture_movie import init_model
from architectures.model_architecture_simple import init_model
#from architectures.model_architecture_cp import init_model
#from architectures.highway import init_model

free_all_buffers()
N_FRAMES = 3

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()[:4]

F1_IND = 0
F2_IND = find_layer(LAYERS,'F2')
F3_IND = find_layer(LAYERS,'F3_lin')

ERR_IND = find_layer(LAYERS, 'ERR')
F1c_IND = find_layer(LAYERS, 'F1c')
F2c_IND = find_layer(LAYERS, 'F2c')
x1t = random_function(np.concatenate(((N_FRAMES,), LAYERS[F1_IND]['in_shape'][1]))) * 1e-1
set_buffer(random_function(LAYERS[ERR_IND]['in_shape'][1]), WEIGHTS[ERR_IND][1]) # target
#set_buffer(random_function(LAYERS[F1c_IND]['in_shape'][1]), WEIGHTS[F1c_IND][1]) # target
#set_buffer(random_function(LAYERS[F2c_IND]['in_shape'][1]), WEIGHTS[F2c_IND][1]) # target
set_buffer(random_function(LAYERS[F2_IND]['in_shape'][1]), WEIGHTS[F2_IND][1]) # target
#set_buffer(random_function(LAYERS[F3_IND]['in_shape'][1]), WEIGHTS[F3_IND][1]) # target


OUTPUT = None; WEIGHT_DERIVS = None
MEM_DERIVS = [None]*len(MEM_INDS)
OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)

PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
frame = 0
set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs

OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)

# update partials_prev
MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)

OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)

#sz = return_buffer(WEIGHT_DERIVS[gradient_layer][gradient_arg]).ravel()[i_ind]

