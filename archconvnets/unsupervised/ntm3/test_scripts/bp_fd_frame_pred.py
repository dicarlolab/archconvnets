import numpy as np
import time
import scipy.optimize
from scipy.io import loadmat
from ntm_core import *
from archconvnets.unsupervised.ntm3.architectures.ctt_frame_pred_test_fd import *
from scipy.stats import pearsonr

free_all_buffers()

scalar = 1e-1
################ init save vars

frame = 0; frame_local = 0; err = 0; corr = 0; movie_ind = 0; 
cifar_err = 0; cifar_class = 0
cat_err = 0; cat_class = 0; obj_err = 0; obj_class = 0

N_FUTURE = 1 # how far into the future to predict
EPOCH_LEN = 11 # length of movie
SAVE_FREQ = 10000/(BATCH_SZ) # instantaneous checkpoint
WRITE_FREQ = 10000/(BATCH_SZ) # new checkpoint
FRAME_LAG = 100 #SAVE_FREQ

target_buffer = np.zeros((SAVE_FREQ, N_IN),dtype='single')
output_buffer = np.zeros((SAVE_FREQ, N_IN),dtype='single')

err_log = []; corr_log = []; cifar_err_log = []; cifar_class_log = []
cat_err_log = []; cat_class_log = []; obj_err_log = []; obj_class_log = []

t_start = time.time()

################ init weights and inputs
LAYERS, WEIGHTS, MEM_INDS, PREV_VALS, print_names = init_model()

CIFAR_PRED_IND = find_layer(LAYERS, 'CIFAR')
CIFAR_DIFF_IND = find_layer(LAYERS, 'CIFAR_ERR')
CIFAR_OUT_IND = find_layer(LAYERS, 'CIFAR_SUM_ERR')

F1_IND = 0

MEM_DERIVS = [None]*len(MEM_INDS)

OUTPUT = None; WEIGHT_DERIVS = None; WEIGHT_DERIVS_RMS = None
OUTPUT_CIFAR = None; WEIGHT_DERIVS_CIFAR = None; WEIGHT_DERIVS_RMS_CIFAR = None
OUTPUT_OBJ = None; WEIGHT_DERIVS_OBJ = None; WEIGHT_DERIVS_RMS_OBJ = None
OUTPUT_CAT = None; WEIGHT_DERIVS_CAT = None; WEIGHT_DERIVS_RMS_CAT = None

OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
PARTIALS_PREV = init_partials(LAYERS, MEM_INDS)

#set_buffer(cifar_inputs, WEIGHTS[F1_IND][1])
set_buffer(random_function(LAYERS[F1_IND]['in_shape'][1]), WEIGHTS[F1_IND][1])

set_buffer(random_function(LAYERS[CIFAR_DIFF_IND]['in_shape'][1]), WEIGHTS[CIFAR_DIFF_IND][1]) # target


OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
print pearsonr(return_buffer(OUTPUT[CIFAR_PRED_IND])[0].squeeze(), return_buffer(WEIGHTS[CIFAR_DIFF_IND][1])[0].squeeze())
print return_buffer(OUTPUT[CIFAR_DIFF_IND])[0]

################ which gradient to test
gradient_layer = 0#F1_IND
gradient_arg = 0

def f(y):
	OUTPUT = None
	OUTPUT_PREV = init_output_prev(LAYERS, MEM_INDS, PREV_VALS)
	Wy = return_buffer(WEIGHTS[gradient_layer][gradient_arg])
	weights_shape = Wy.shape; Wy = Wy.ravel(); Wy[i_ind] = y
	set_buffer(Wy.reshape(weights_shape), WEIGHTS[gradient_layer][gradient_arg])
	
	###############
	# forward
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	z = return_buffer(OUTPUT[CIFAR_OUT_IND])[0]
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
	
	###############
	# forward
	OUTPUT = forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV)
	
	# reverse
	WEIGHT_DERIVS = reverse_network(CIFAR_OUT_IND, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS_PREV, WEIGHT_DERIVS)
	
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
