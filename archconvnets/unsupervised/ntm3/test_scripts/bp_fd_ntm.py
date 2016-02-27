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
F12_IND = find_layer(LAYERS,'F12')
X1_IND = find_layer(LAYERS,'FL_lin')
X2_IND = find_layer(LAYERS,'T2_lin')
X3_IND = find_layer(LAYERS,'T3_lin')
X4_IND = find_layer(LAYERS,'T4_lin')

X1b_IND = find_layer(LAYERS,'T1_b')

ERR_IND = find_layer(LAYERS, 'ERR')
x1t = random_function(np.concatenate(((N_FRAMES,), LAYERS[F1_IND]['in_shape'][1]))) * 1e0
set_buffer(random_function(LAYERS[ERR_IND]['in_shape'][1]), WEIGHTS[ERR_IND][1]) # target

#set_buffer(random_function(LAYERS[X1_IND]['in_shape'][1]), WEIGHTS[X1_IND][1]) # target
#set_buffer(random_function(LAYERS[X2_IND]['in_shape'][1]), WEIGHTS[X2_IND][1]) # target
#set_buffer(random_function(LAYERS[X3_IND]['in_shape'][1]), WEIGHTS[X3_IND][1]) # target
#set_buffer(random_function(LAYERS[X4_IND]['in_shape'][1]), WEIGHTS[X4_IND][1]) # target
#set_buffer(random_function(LAYERS[X5_IND]['in_shape'][1]), WEIGHTS[X5_IND][1]) # target

################ which gradient to test
gradient_layer = F12_IND
gradient_arg = 0

def f(y):
	OUTPUT = None
	OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
	
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	for frame in range(N_FRAMES):
		set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs
		set_buffer(x1t[frame], WEIGHTS[F12_IND][1])  # inputs
		
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	z = return_buffer(OUTPUT[-1])[0]
	free_list(OUTPUT)
	free_list(OUTPUT_PREV)
	return z

def g(y):
	OUTPUT = None; WEIGHT_DERIVS = None; MEM_DERIVS = [None]*len(MEM_INDS); OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS); PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	for frame in range(N_FRAMES):
		set_buffer(x1t[frame], WEIGHTS[F1_IND][1])  # inputs
		set_buffer(x1t[frame], WEIGHTS[F12_IND][1])  # inputs
		
		OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
		WEIGHT_DERIVS = reverse_network(len(LAYERS)-1, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
		
		# update partials_prev
		MEM_DERIVS = reverse_network(MEM_INDS, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, MEM_DERIVS, keep_dims=True)
		PARTIALS_PREV = copy_partials(MEM_INDS, LAYERS, PARTIALS_PREV, MEM_DERIVS)
		
		OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
		
	z = return_buffer(WEIGHT_DERIVS[gradient_layer][gradient_arg]).ravel()[i_ind]
	
	free_list_list(MEM_DERIVS)
	free_partials(PARTIALS_PREV)
	free_list(OUTPUT)
	free_list(WEIGHT_DERIVS)
	free_list(OUTPUT_PREV)
	return z

assert isinstance(LAYERS[gradient_layer]['in_source'][gradient_arg], int) != True, 'derivative of intermediate layer'
ref = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e5

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
t_start = time.time()
for sample in range(N_SAMPLES):
	i_ind = np.random.randint(np.prod(ref.shape))
	y = ref.ravel()[i_ind]
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std(), time.time() - t_start, GPU
