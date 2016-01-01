import numpy as np
import copy
import time
import scipy.optimize
from ntm_core_gpu import *
from archconvnets.unsupervised.ntm_module2.ntm_module2 import *

N_FRAMES = 5
N_CONTROLLERS = 16
M_LENGTH = 6
N_MEM_SLOTS = 8
mem_shape = (N_MEM_SLOTS, M_LENGTH)

free_all_buffers()

#############
LAYERS = []

# F under
FU_IND = len(LAYERS)
x_shape = (12,M_LENGTH)
F0_shape = (8, 12)
L0_shape = (8,M_LENGTH)
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L0_shape, \
				'in_shape': [F0_shape, x_shape], \
				'in_source': [random_function, -1], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# FR read
FR_IND = len(LAYERS)
F1_shape = (N_CONTROLLERS,8)
L1_shape = (F1_shape[0], x_shape[1])
LAYERS.append({ 'forward_F': linear_F, \
				'out_shape': L1_shape, \
				'in_shape': [F1_shape, L0_shape], \
				'in_source': [random_function, FU_IND], \
				'deriv_F': [linear_F_dF, linear_F_dx] })

# sum
LAYERS.append({ 'forward_F': sum_points, \
				'out_shape': (1,), \
				'in_shape': [L1_shape], \
				'in_source': [FR_IND], \
				'deriv_F': [sum_points_dinput] })				

################
WEIGHTS = init_weights(LAYERS)
xt = random_function(np.concatenate(((N_FRAMES,), LAYERS[FU_IND]['in_shape'][1])))
set_buffer(xt[0], WEIGHTS[FU_IND][1])
check_weights(WEIGHTS, LAYERS)

OUTPUT_PREV = [None] * len(LAYERS)
#OUTPUT_PREV[MEM_IND] = init_buffer(random_function(LAYERS[MEM_IND]['out_shape']))
check_output_prev(OUTPUT_PREV, LAYERS)

OUTPUT = None

OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)

