import numpy as np
from scipy.io import loadmat
from ntm_core import *
import Image
import random
from archconvnets.unsupervised.rosch_models_collated_reduced import *

def resize_inputs(inputs): # 2500,3,32,32
	inputs_resized = np.zeros((inputs.shape[0], 3, IM_SZ_R, IM_SZ_R), dtype='single')
	for img in range(inputs.shape[0]):
		inputs_resized[img] = np.asarray(Image.fromarray(inputs[img].transpose((1,2,0))).resize((IM_SZ_R,IM_SZ_R))).transpose((2,0,1))
	return np.single(inputs_resized[:,np.newaxis])/255 - .5

#############################
# movies
N_MOVIES = 539
MOVIE_FILE_SZ = 2500
N_FILES_TEST_MOVIE = 1
N_CAT_MOVIE = len(np.unique(syn_cats))
N_OBJ_MOVIE = 91

N_TEST = 500
N_BATCHES_TEST_MOVIE = N_TEST / BATCH_SZ

z = loadmat('/home/darren/new_movies3_cut/0.mat')

movie_test_inputs = np.single(z['inputs'])/255 - .5
movie_test_base_frame = resize_inputs(z['inputs'][:,-1])

movie_test_targets = np.ascontiguousarray((np.single(z['targets'])/255 - movie_test_base_frame)[:, :N_FUTURE])

movie_test_objs = z['obj_list'].squeeze()

l = np.zeros((MOVIE_FILE_SZ, N_CAT_MOVIE),dtype='uint8')
l[np.arange(MOVIE_FILE_SZ), syn_cats[movie_test_objs]] = 1
Y_test_movie_cat = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

l = np.zeros((MOVIE_FILE_SZ, N_OBJ_MOVIE),dtype='uint8')
l[np.arange(MOVIE_FILE_SZ), movie_test_objs] = 1
Y_test_movie_obj = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

movie_train_objs = []
movie_train_inputs = []
movie_train_base_frame = []
movie_train_targets = []
Y_train_movie_cat = []
Y_train_movie_obj = []

def load_movie_seqs(batch, frame, CAT_DIFF_IND, OBJ_DIFF_IND, DIFF_IND, F1_IND, PX_IND, WEIGHTS, testing=False):
	global movie_train_objs, Y_train_movie_cat, Y_train_movie_obj, movie_train_inputs, movie_train_targets, movie_train_base_frame

	movie_batch = batch % (MOVIE_FILE_SZ / BATCH_SZ)
	
	if testing:
		obj_target = Y_test_movie_obj[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		cat_target = Y_test_movie_cat[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		
		objs = movie_test_objs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		cats = syn_cats[movie_test_objs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]]
		
		movie_targets = movie_test_targets[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		movie_inputs = movie_test_inputs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
	else:
		
		if movie_batch == 0:
			movie_file = ((batch*BATCH_SZ)/MOVIE_FILE_SZ) % (N_MOVIES - N_FILES_TEST_MOVIE)
			
			z = loadmat('/home/darren/new_movies3_cut/' + str(movie_file + N_FILES_TEST_MOVIE) + '.mat')
			
			movie_train_inputs = np.single(z['inputs'])/255 - .5
			movie_train_base_frame = resize_inputs(z['inputs'][:,-1])
			
			movie_train_targets = np.ascontiguousarray((np.single(z['targets'])/255 - movie_train_base_frame)[:, :N_FUTURE])
			
			movie_train_objs = z['obj_list'].squeeze()

			l = np.zeros((MOVIE_FILE_SZ, N_CAT_MOVIE),dtype='uint8')
			l[np.arange(MOVIE_FILE_SZ), syn_cats[movie_train_objs]] = 1
			Y_train_movie_cat = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

			l = np.zeros((MOVIE_FILE_SZ, N_OBJ_MOVIE),dtype='uint8')
			l[np.arange(MOVIE_FILE_SZ), movie_train_objs] = 1
			Y_train_movie_obj = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories
			
		#############
		
		obj_target = Y_train_movie_obj[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		cat_target = Y_train_movie_cat[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		
		objs = movie_train_objs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		cats = syn_cats[movie_train_objs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]]
		
		movie_targets = movie_train_targets[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		movie_inputs = movie_train_inputs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		
	
	cat_target = np.ascontiguousarray(cat_target)
	obj_target = np.ascontiguousarray(obj_target)
	
	set_buffer(cat_target, WEIGHTS[CAT_DIFF_IND][1])
	set_buffer(obj_target, WEIGHTS[OBJ_DIFF_IND][1])
	
	if frame >= 3: # test phase
		px_input = np.ascontiguousarray(movie_inputs[:,2])
		target = movie_targets[:,frame-3]
	else: # input phase
		px_input = np.ascontiguousarray(movie_inputs[:,frame])
		target = movie_targets[:,0]
		
	set_buffer(np.ascontiguousarray(target.reshape((BATCH_SZ, np.prod(target.shape[1:]), 1))), WEIGHTS[DIFF_IND][1])
	set_buffer(px_input, WEIGHTS[F1_IND][1])
	set_buffer(px_input, WEIGHTS[PX_IND][1])
	
	#print px_input.shape, np.ascontiguousarray(target.reshape((BATCH_SZ, np.prod(target.shape[1:]), 1))).shape, testing
	
	return objs,cats, cat_target, obj_target, movie_inputs, movie_targets
	
