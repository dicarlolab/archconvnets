from gpu_flag import *
import numpy as np
import copy
import time
from scipy.io import savemat
from archconvnets.unsupervised.ntm_module3.ntm_module3 import *

def check_weights(WEIGHTS, LAYERS):
	check_network(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_shape'])
		
		for arg in range(N_ARGS):
			if isinstance(L['in_source'][arg], int) == False or L['in_source'][arg] == -1:
				assert WEIGHTS[layer_ind][arg] is not None, 'layer %i argument %i not initialized' % (layer_ind, arg)
				assert WEIGHTS[layer_ind][arg][1] == L['in_shape'][arg], 'layer %s argument %i not initialized to right size' % (L['name'], arg)
			else:
				assert WEIGHTS[layer_ind][arg] is None, 'layer %i argument %i should not have weightings because it should be computed from layer %i' % (layer_ind, arg,  L['in_source'][arg])


def check_network(LAYERS):
	n_allocated = return_n_allocated()
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		assert isinstance(L['name'],str)
		assert L['name'] != '-' # reserved for prior time step layer outputs
		assert isinstance(L['out_shape'],tuple)
		assert len(L['in_prev']) == len(L['in_shape']) == len(L['deriv_F']) == len(L['in_source'])
		assert len(L['additional_deriv_args']) == len(L['deriv_F'])
		
		# build arguments
		N_ARGS = len(L['in_shape'])
		args = [None] * N_ARGS
		for arg in range(N_ARGS):
			args[arg] = init_buffer(np.asarray(np.random.random(L['in_shape'][arg]),dtype='single'))
		
		# check if function corretly produces specified output dimensions
		LAYER_OUT = L['forward_F'](args, additional_args=L['additional_forward_args'])
		assert return_buffer(LAYER_OUT).shape == L['out_shape'], "layer %s (%i) didn't produce expected output (%i, %i)" % (L['name'], layer_ind, np.prod(LAYER_OUT[1]), np.prod(L['out_shape']))
		
		# check if deriv functions correctly produce correct shapes
		DERIV_ABOVE = init_buffer(np.zeros(np.concatenate(((2,3), L['out_shape'])), dtype='single'))
		for arg in range(N_ARGS):
			expected_shape = tuple(np.concatenate(((2,3), L['in_shape'][arg])))
			OUT = L['deriv_F'][arg](args, LAYER_OUT, DERIV_ABOVE, additional_args=L['additional_deriv_args'][arg])

			#print L['name'],arg
			assert return_buffer(OUT).shape == expected_shape, 'deriv not expected size (layer %s, arg %i)' % (L['name'], arg)
			
			free_buffer(OUT)
		free_buffer(LAYER_OUT)
		free_buffer(DERIV_ABOVE)
		
		# free mem
		for arg in range(N_ARGS):
			free_buffer(args[arg])
		
		# check if other layers claim to produce expected inputs
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_shape'][arg] == LAYERS[L['in_source'][arg]]['out_shape'], '%s %i' % (L['name'], arg)
				
		# check if layers are ordered (no inputs to this layer come after this one in the list... unless recursive mem layer)
		for arg in range(N_ARGS):
			if L['in_source'][arg] >= 0 and isinstance(L['in_source'][arg], int):
				assert L['in_source'][arg] < layer_ind or L['in_prev'][arg]
	assert n_allocated == return_n_allocated(), 'check_network() leaked memory'

def check_output_prev(OUTPUT_PREV, LAYERS):
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		if layer_ind in L['in_source']:
			assert OUTPUT_PREV[layer_ind][1] == L['out_shape']

def init_weights(LAYERS):
	check_network(LAYERS)
	WEIGHTS = [None]*len(LAYERS)
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		N_INPUTS = len(L['in_shape'])
		WEIGHTS[layer_ind] = [None]*N_INPUTS
		for arg in range(N_INPUTS):
			if isinstance(L['in_source'][arg], int) != True:
				WEIGHTS[layer_ind][arg] = init_buffer(L['in_source'][arg]( L['in_shape'][arg] ))
			elif L['in_source'][arg] == -1: # user supplied
				WEIGHTS[layer_ind][arg] = init_buffer()
				
	return WEIGHTS

	
def build_forward_args(L, layer_ind, OUTPUT, OUTPUT_PREV, WEIGHTS):
	N_ARGS = len(L['in_shape'])
	args = [None] * N_ARGS
	
	for arg in range(N_ARGS):
		src = L['in_source'][arg]
		
		# input is from another layer
		if isinstance(src, int) and src != -1:
			if L['in_prev'][arg]: # from prior timestep
				args[arg] = OUTPUT_PREV[src]
			else: # from current timestep
				args[arg] = OUTPUT[src]
		else: # input is a weighting
			args[arg] = WEIGHTS[layer_ind][arg]
		
	return args

def forward_network(LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, break_layer=None):
	#check_weights(WEIGHTS, LAYERS)
	#check_output_prev(OUTPUT_PREV, LAYERS)
	
	OUTPUT = init_gpu_list(OUTPUT, LAYERS, args=False)
	
	if break_layer is not None:
		n_layers = break_layer+1
	else:
		n_layers = len(LAYERS)
	
	for layer_ind in range(n_layers):
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_shape'])

		args = build_forward_args(L, layer_ind, OUTPUT, OUTPUT_PREV, WEIGHTS)
		L['forward_F'](args, OUTPUT[layer_ind], additional_args=L['additional_forward_args'])
		
	return OUTPUT
	
def init_gpu_list(LIST, LAYERS, args=True):
	if LIST is None:
		LIST = [None] * len(LAYERS)
		for layer_ind in range(len(LAYERS)):
			L = LAYERS[layer_ind]
			N_ARGS = len(L['in_shape'])
			
			# buffers for each layer's args
			if args:
				LIST[layer_ind] = [None]*N_ARGS
				for arg in range(N_ARGS):
					LIST[layer_ind][arg] = init_buffer()
					
			# buffer only for each layer (ex. layer outputs)
			else:
				LIST[layer_ind] = init_buffer()
	return LIST


# apply chain-rule down the network
def reverse_network(layer_ind, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS, WEIGHT_DERIVS, keep_dims=False, scalar=1, reset_derivs=True, abort_layer=None): # multiply all partials together
	if abort_layer is not None:
		abort_layer = find_layer(LAYERS, abort_layer)
	
	# compute partials starting from single layer
	if isinstance(layer_ind,int):
		WEIGHT_DERIVS = init_gpu_list(WEIGHT_DERIVS, LAYERS)
		if reset_derivs:
			zero_buffer_list(WEIGHT_DERIVS)

		WEIGHT_DERIVS = reverse_network_recur(None, layer_ind, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS, WEIGHT_DERIVS, keep_dims, scalar, abort_layer)
	# compute partials from multiple layers, store result in a list
	else:
		for i in range(len(layer_ind)):
			WEIGHT_DERIVS[i] = init_gpu_list(WEIGHT_DERIVS[i], LAYERS)
			if reset_derivs:
				zero_buffer_list(WEIGHT_DERIVS[i])

			WEIGHT_DERIVS[i] = reverse_network_recur(None, layer_ind[i], LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS, WEIGHT_DERIVS[i], keep_dims, scalar, abort_layer)
	return WEIGHT_DERIVS

def reverse_network_recur(deriv_above, layer_ind, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS, WEIGHT_DERIVS, keep_dims, scalar, abort_layer): # multiply all partials together
	L = LAYERS[layer_ind]
	N_ARGS = len(L['in_shape'])
	deriv_above_created = False
	
	if deriv_above is None:
		deriv_above_shape = np.concatenate((LAYERS[layer_ind]['out_shape'], LAYERS[layer_ind]['out_shape']))
		deriv_above = init_buffer(np.single(np.eye(np.prod(LAYERS[layer_ind]['out_shape'])).reshape(deriv_above_shape)))
		deriv_above_created = True
	
	for arg in range(N_ARGS):
		src = L['in_source'][arg]
		if L['in_source'][arg] != -1: # don't compute gradients for user-supplied entries (Ex. images)
			
			# compute derivs
			args = build_forward_args(L, layer_ind, OUTPUT, OUTPUT_PREV, WEIGHTS)
			deriv_above_new = L['deriv_F'][arg](args, OUTPUT[layer_ind], deriv_above, additional_args=L['additional_deriv_args'][arg])
			
			# input is a layer:
			if isinstance(src, int) and src != -1:
				# memory partials, stop here, add these partials to the correct weight derivs:
				if L['in_prev'][arg]:
					P = PARTIALS[src]
					N_ARGS2 = len(P['in_source'])
					for arg2 in range(N_ARGS2):
						p_layer_ind = P['in_source'][arg2]
						p_arg = P['in_arg'][arg2]
						p_partial = P['partial'][arg2]
						
						deriv_temp = mult_partials(deriv_above_new, p_partial, LAYERS[src]['out_shape'])
						WEIGHT_DERIVS[p_layer_ind][p_arg] = add_points_inc((WEIGHT_DERIVS[p_layer_ind][p_arg], deriv_temp), scalar=scalar)
						
						squeeze_dim1(WEIGHT_DERIVS[p_layer_ind][p_arg], keep_dims)
						free_buffer(deriv_temp)
						
				# another layer (At this time step, go back to earlier layers)
				# [do not go back if this is the abort_layer where we stop backpropping]
				elif src != abort_layer:
					reverse_network_recur(deriv_above_new, src, LAYERS, WEIGHTS, OUTPUT, OUTPUT_PREV, PARTIALS, WEIGHT_DERIVS, keep_dims, scalar, abort_layer)
			
			# input is not a layer, end here
			else:
				WEIGHT_DERIVS[layer_ind][arg] = add_points_inc((WEIGHT_DERIVS[layer_ind][arg], deriv_above_new), scalar=scalar)
				squeeze_dim1(WEIGHT_DERIVS[layer_ind][arg], keep_dims)
				
			free_buffer(deriv_above_new)
	
	if deriv_above_created:
		free_buffer(deriv_above)
	
	return WEIGHT_DERIVS

def init_traverse_to_end(layer_orig, layer_cur, arg, LAYERS, PARTIALS):
	dest = LAYERS[layer_cur]['in_source'][arg]
	
	# don't traverse previous states
	if LAYERS[layer_cur]['in_prev'][arg] == False:
		
		# input or weights, end:
		if (isinstance(dest, int) == False) or dest == -1:
			# have these inputs already been added?
			t1 = np.asarray(PARTIALS[layer_orig]['in_source'])
			t2 = np.asarray(PARTIALS[layer_orig]['in_arg'])
			partials_added = np.sum((t1 == layer_cur) * (t2 == arg)) #inds = np.nonzero((t1 == layer_cur) * (t2 == arg))[0]
			
			#assert partials_added <= 1, 'partials have been added more than once'
			
			# inputs have not been added, add them:
			if partials_added == 0:
				PARTIALS[layer_orig]['in_source'].append(layer_cur)
				PARTIALS[layer_orig]['in_arg'].append(arg)
				OUT = init_buffer(np.zeros(np.concatenate((LAYERS[layer_orig]['out_shape'], LAYERS[layer_cur]['in_shape'][arg])), dtype='single'))
				PARTIALS[layer_orig]['partial'].append(OUT)
		
		# another layer, go back farther through the network:
		else:
			N_ARGS2 = len(LAYERS[dest]['in_source'])
			for arg2 in range(N_ARGS2):
				init_traverse_to_end(layer_orig, dest, arg2, LAYERS, PARTIALS)

# collect all weight partials which contribute to the memory layers.
# store them at the memory layer
def init_partials(LAYERS, MEM_INDS):
	PARTIALS = [None]*len(LAYERS)
	
	for layer_ind in MEM_INDS:
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_source'])
		PARTIALS[layer_ind] = {'in_source': [], 'in_arg': [], 'partial': []}
		
		for arg in range(N_ARGS):
			if L['in_prev'][arg] == False:
				init_traverse_to_end(layer_ind, layer_ind, arg, LAYERS, PARTIALS)
			
	return PARTIALS
	
def free_partials(PARTIALS_PREV):
	for layer_ind in range(len(PARTIALS_PREV)):
		if PARTIALS_PREV[layer_ind] is not None:
			free_list(PARTIALS_PREV[layer_ind]['partial'])

def copy_traverse_to_end(layer_orig, layer_cur, arg, LAYERS, PARTIALS, MEM_WEIGHT_DERIVS):
	dest = LAYERS[layer_cur]['in_source'][arg]
	
	if LAYERS[layer_cur]['in_prev'][arg] == False:
		# end (weighting, input or mem layer input):
		if (isinstance(dest, int) == False) or dest == -1:
			# have these inputs already been added?
			t1 = np.asarray(PARTIALS[layer_orig]['in_source'])
			t2 = np.asarray(PARTIALS[layer_orig]['in_arg'])
			inds = np.nonzero((t1 == layer_cur) * (t2 == arg))[0]
			
			#assert len(inds) == 1, 'partials have not been added to partials list %i' % len(inds)
			
			# copy partials to mem_weight_derivs
			# note: there is redundant copying happening if a layer contributes to multiple
			# branches...this in principle should be checked for to save some time
			copy_buffer(MEM_WEIGHT_DERIVS[layer_cur][arg], PARTIALS[layer_orig]['partial'][inds[0]])
		
		# continue (another layer)
		else:
			N_ARGS2 = len(LAYERS[dest]['in_source'])
			for arg2 in range(N_ARGS2):
				PARTIALS = copy_traverse_to_end(layer_orig, dest, arg2, LAYERS, PARTIALS, MEM_WEIGHT_DERIVS)
	return PARTIALS

# copy MEM_DERIVS (list of partials starting at memory layer LAYER_IND[i]):
# into PARTIALS_PREV at the memory layer entry LAYER_IND[i] in PARTIALS_PREV
def copy_partials(LAYER_IND, LAYERS, PARTIALS_PREV, MEM_DERIVS):
	#assert len(LAYER_IND) == len(MEM_DERIVS)
	for i in range(len(LAYER_IND)):
		layer_ind = LAYER_IND[i]
		L = LAYERS[layer_ind]
		N_ARGS = len(L['in_source'])
		
		for arg in range(N_ARGS):
			if L['in_prev'][arg] == False:
				PARTIALS_PREV = copy_traverse_to_end(layer_ind, layer_ind, arg, LAYERS, PARTIALS_PREV, MEM_DERIVS[i])
	return PARTIALS_PREV


def find_layer(LAYERS, name):
	if isinstance(name, str):
		if name[-1] == '-':
			name = name[:len(name)-1]
		for layer_ind in range(len(LAYERS)):
			if LAYERS[layer_ind]['name'] == name:
				return layer_ind
	else:
		INDS = [None]*len(name)
		for i in range(len(name)):
			for layer_ind in range(len(LAYERS)):
				if LAYERS[layer_ind]['name'] == name[i]:
					INDS[i] = layer_ind
		return INDS
	return None
	
# randomly generate outputs for layer INDS
def random_function_list(LAYERS, INDS):
	PREV_VALS = []
	for layer_ind in INDS:
		PREV_VALS.append(random_function(LAYERS[layer_ind]['out_shape']))
	return PREV_VALS

# move PREV_VALS into OUTPUT_PREV in layers INDS
def init_output_prev(LAYERS, INDS, PREV_VALS):
	OUTPUT_PREV = [None]*len(LAYERS)
	for layer_ind in range(len(INDS)):
		#assert OUTPUT_PREV[INDS[layer_ind]] is None
		OUTPUT_PREV[INDS[layer_ind]] = init_buffer(PREV_VALS[layer_ind])
	
	return OUTPUT_PREV


def update_weights(LAYERS, WEIGHTS, WEIGHT_DERIVS, EPS):
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		for arg in range(len(L['in_source'])):
			# only update weight layers, not input layers
			if hasattr(L['in_source'][arg], '__call__') and WEIGHT_DERIVS[layer_ind][arg][1] is not None: # if there was a deriv computed:
				add_points_inc((WEIGHTS[layer_ind][arg], WEIGHT_DERIVS[layer_ind][arg]), scalar=EPS)
	return WEIGHTS
	
def update_weights_rms(LAYERS, WEIGHTS, WEIGHT_DERIVS, WEIGHT_DERIVS_RMS, EPS, frame, FRAME_LAG):
	WEIGHT_DERIVS_RMS = init_gpu_list(WEIGHT_DERIVS_RMS, LAYERS)
	deriv_sq = None
	
	for layer_ind in range(len(LAYERS)):
		L = LAYERS[layer_ind]
		for arg in range(len(L['in_source'])):
			# only update weight layers, not input layers
			if hasattr(L['in_source'][arg], '__call__'):
				#assert WEIGHT_DERIVS[layer_ind][arg][1] is not None, 'layer %s (%i), arg %i has no gradient. is its output connected to the rest of the network?' % \
				#	(LAYERS[layer_ind]['name'], layer_ind, arg)
				
				if WEIGHT_DERIVS[layer_ind][arg][1] is not None: # if there was a deriv computed
					
					# deriv_sq = WEIGHT_DERIVS ** 2
					deriv_sq = sq_points([WEIGHT_DERIVS[layer_ind][arg]], deriv_sq, deriv_computable=False)
					
					if WEIGHT_DERIVS_RMS[layer_ind][arg][1] is None: # init WEIGHT_DERIVS_RMS, todo: cleanup this
						copy_buffer(WEIGHT_DERIVS[layer_ind][arg], WEIGHT_DERIVS_RMS[layer_ind][arg])
						zero_buffer(WEIGHT_DERIVS_RMS[layer_ind][arg])
					
					# WEIGHT_DERIVS_RMS = 0.9*WEIGHT_DERIVS_RMS + 0.1*WEIGHT_DERIVS**2
					add_points_inc((WEIGHT_DERIVS_RMS[layer_ind][arg], deriv_sq), scalar0=.9, scalar=.1)
					deriv_sq = free_buffer(deriv_sq)
					
					# WEIGHT_DERIVS /= sqrt(WEIGHT_DERIVS_RMS
					point_wise_div_sqrt((WEIGHT_DERIVS[layer_ind][arg], WEIGHT_DERIVS_RMS[layer_ind][arg]), clip=10)
					
					if frame > FRAME_LAG:
						add_points_inc((WEIGHTS[layer_ind][arg], WEIGHT_DERIVS[layer_ind][arg]), scalar=EPS)
				
	return WEIGHT_DERIVS_RMS

def print_layer(LAYERS, print_name, WEIGHTS, WEIGHT_DERIVS, OUTPUT, max_print_len, EPS):
	w_ind = find_layer(LAYERS, print_name+'_lin')
	b_ind = find_layer(LAYERS, print_name+'_b')
	o_ind = find_layer(LAYERS, print_name)
	if b_ind is None: # if there are no linearities (sigmoid, relu)
		b_ind = o_ind
	
	O = return_buffer(OUTPUT[o_ind])

	if w_ind is None:
		print '  ', print_name, ' '*(max_print_len - len(print_name)), '  %.1e %.1e %.1e' % (np.min(O), np.median(O), np.max(O))
	else:
		W = return_buffer(WEIGHTS[w_ind][0])
		B = return_buffer(WEIGHTS[b_ind][1])
		
		DW = return_buffer(WEIGHT_DERIVS[w_ind][0])
		DB = return_buffer(WEIGHT_DERIVS[b_ind][1])
		
		print '  ', print_name, ' '*(max_print_len - len(print_name)), \
				' W: %.1e %.1e (%.1e)  B: %.1e %.1e (%.1e) -- %.1e %.1e' % (\
			np.min(W), np.max(W), -EPS*np.median(np.abs(DW/W)), \
			np.min(B), np.max(B), -EPS*np.median(np.abs(DB/B)), np.min(O), np.max(O))


def print_state(LAYERS, WEIGHTS, WEIGHT_DERIVS, OUTPUT, EPS, err_log, frame, corr_log, cifar_err_log, cifar_class_log, imgnet_err_log, imgnet_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, t_start, save_name, print_names):
	print 'frame: ', frame, 'time: ', time.time() - t_start, 'GPU:', GPU_IND, save_name
	print 'imgnet_class: ', imgnet_class_log[-1], 'imgnet_err: ', imgnet_err_log[-1]
	print 'cifar_class: ', cifar_class_log[-1], 'cifar_err: ', cifar_err_log[-1]
	print 'obj_class: ', obj_class_log[-1], 'obj_err: ', obj_err_log[-1]
	print 'cat_class: ', cat_class_log[-1], 'cat_err: ', cat_err_log[-1]
	
	max_print_len = 0
	for print_name in print_names:
		if len(print_name) > max_print_len:
			max_print_len = len(print_name)
	
	for print_name in print_names:
		if len(print_name) != 0: # print layer
			if print_name[0] != '_': # standard layer
				print_layer(LAYERS, print_name, WEIGHTS, WEIGHT_DERIVS, OUTPUT, max_print_len, EPS)
			else: # read/write layers
				print_layer(LAYERS, 'R'+print_name, WEIGHTS, WEIGHT_DERIVS, OUTPUT, max_print_len, EPS)
				print_layer(LAYERS, 'W'+print_name, WEIGHTS, WEIGHT_DERIVS, OUTPUT, max_print_len, EPS)
		else: # print blank
			print
	print '---------------------'

def save_state(LAYERS, WEIGHTS, WEIGHTS_F1_INIT, save_name, output_buffer, target_buffer, EPS, err_log, corr_log, EPOCH_LEN):
	WEIGHTS_F1 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1_lin')][0])
	WEIGHTS_F2 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F2_lin')][0])
	WEIGHTS_F3 = return_buffer(WEIGHTS[find_layer(LAYERS, 'FL_lin')][0])
	savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'target_buffer': target_buffer, \
			'err_log': err_log, 'corr_log': corr_log, 'EPS': EPS, \
			'F1_init': WEIGHTS_F1_INIT, 'F1': WEIGHTS_F1, 'F2': WEIGHTS_F2, 'F3': WEIGHTS_F3, 'EPOCH_LEN': EPOCH_LEN})

def save_conv_state(LAYERS, WEIGHTS, WEIGHTS_F1_INIT, save_name, movie_inputs, output_buffer, target_buffer, EPS, err_log, corr_log, cifar_err_log, cifar_class_log, imgnet_err_log, imgnet_class_log, obj_err_log, obj_class_log, cat_err_log, cat_class_log, EPOCH_LEN, N_MOVIES):
	WEIGHTS_F1 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F1')][0])
	WEIGHTS_F2 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F2')][0])
	WEIGHTS_F3 = return_buffer(WEIGHTS[find_layer(LAYERS, 'F3')][0])
	savemat('/home/darren/' + save_name + '.mat', {'output_buffer': output_buffer, 'target_buffer': target_buffer, 'N_MOVIES': N_MOVIES, \
			'err_log': err_log, 'corr_log': corr_log, 'cifar_err_log': cifar_err_log, 'cifar_class_log': cifar_class_log, 'EPS': EPS, \
			'imgnet_err_log': imgnet_err_log, 'imgnet_class_log': imgnet_class_log, 'movie_inputs': movie_inputs, \
			'cat_err_log': cat_err_log, 'cat_class_log': cat_class_log, 'obj_err_log': obj_err_log, 'obj_class_log': obj_class_log,\
			'F1_init': WEIGHTS_F1_INIT, 'F1': WEIGHTS_F1, 'F2': WEIGHTS_F2, 'F3': WEIGHTS_F3, 'EPOCH_LEN': EPOCH_LEN})
