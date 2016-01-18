import numpy as np
import time
import scipy.optimize
from ntm_core import *
from model_architecture_movie_no_mem import init_model
#from model_architecture_movie import init_model
#from model_architecture_simple import init_model
#from model_architecture_cp import init_model

free_all_buffers()
N_FRAMES = 10

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()[:4]

F1_IND = 0
TARGET_IND = find_layer(LAYERS, 'ERR')
x1t = random_function(np.concatenate(((N_FRAMES,), LAYERS[F1_IND]['in_shape'][1]))) / 10
target = random_function(np.concatenate(((N_FRAMES,), LAYERS[TARGET_IND]['in_shape'][1]))) / 10
set_buffer(target[0], WEIGHTS[TARGET_IND][1]) # target

OUTPUT = None; WEIGHT_DERIVS = None
MEM_DERIVS = [None]*len(MEM_INDS)
OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)

PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
for frame in range(N_FRAMES):
	set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
	
	# update partials_prev
	MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
	PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
	
	OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	

