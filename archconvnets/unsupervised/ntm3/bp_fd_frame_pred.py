import numpy as np
import time
import scipy.optimize
from scipy.io import loadmat
from ntm_core import *
#from model_architecture_movie import init_model
#from architectures.model_architecture_simple import init_model
#from architectures.model_architecture_cp import init_model
#from architectures.highway import init_model
from architectures.ctt_frame_pred_test import *

free_all_buffers()
N_MOVIES = 22872 #17750 #12340 #6372
EPOCH_LEN = 11 # length of movie
N_FUTURE = 1 # how far into the future to predict

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS = init_model()[:4]

TARGET_IND = find_layer(LAYERS, 'ERR') # frame target
F1_IND = 0

movie_frame = np.random.randint(EPOCH_LEN - N_CTT - N_FUTURE - N_FRAMES_PRED + 1) + N_CTT # movies

# load movie
movie_name = '/home/darren/rotating_objs32_constback_50t/imgs' + str(np.random.randint(N_MOVIES))  + '.mat'
z = loadmat(movie_name)

inputs = np.ascontiguousarray(np.single(z['imgs'] - .5))

frame_target = (inputs[movie_frame] - inputs[movie_frame+N_FUTURE:movie_frame+N_FUTURE+N_FRAMES_PRED]).ravel()[:,np.newaxis]
frame_target = np.ascontiguousarray(frame_target)

# load targets
set_buffer(frame_target, WEIGHTS[TARGET_IND][1]) # frame target

# forward movie
set_buffer(inputs[movie_frame-N_CTT:movie_frame].reshape((1,N_CTT*3, IM_SZ, IM_SZ)), WEIGHTS[F1_IND][1])  # inputs

################ which gradient to test
gradient_layer = F1_IND
gradient_arg = 0

def f(y):
	OUTPUT = None
	OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
	
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	OUTPUT_PREV = copy_list(OUTPUT, OUTPUT_PREV)
	
	z = return_buffer(OUTPUT[-1])[0]
	free_list(OUTPUT)
	free_list(OUTPUT_PREV)
	return z

def g(y):
	OUTPUT = None; WEIGHT_DERIVS = None
	MEM_DERIVS = [None]*len(MEM_INDS)
	OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)
	
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
eps = np.sqrt(np.finfo(np.float).eps)*1e4

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
