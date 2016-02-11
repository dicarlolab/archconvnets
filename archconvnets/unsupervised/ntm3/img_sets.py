import numpy as np
from scipy.io import loadmat
from gpu_flag import *
import Image
from archconvnets.unsupervised.rosch_models_collated import *

N_BATCHES_TEST = 100

##############################
################## load cifar
N_IMGS_CIFAR = 50000
CIFAR_FILE_SZ = 10000

z2 = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(1))
for batch in range(2,6):
	y = np.load('/home/darren/cifar-10-py-colmajor/data_batch_' + str(batch))
	z2['data'] = np.concatenate((z2['data'], y['data']), axis=1)
	z2['labels'] = np.concatenate((z2['labels'], y['labels']))

z2['data'] = np.single(z2['data'])
z2['data'] /= z2['data'].max()
mean_img_orig = z2['data'].mean(1)[:,np.newaxis]
x = z2['data'] - mean_img_orig
cifar_imgs = np.ascontiguousarray(np.single(x.reshape((3, IM_SZ, IM_SZ, N_IMGS_CIFAR))).transpose((3,0,1,2))[:,np.newaxis])

labels_cifar = np.asarray(z2['labels'])
l = np.zeros((N_IMGS_CIFAR, 10),dtype='uint8')
l[np.arange(N_IMGS_CIFAR),np.asarray(z2['labels']).astype(int)] = 1
Y_cifar = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories
mean_img = mean_img_orig.reshape((1,1,3,IM_SZ,IM_SZ))


# cifar test
z2 = np.load('/home/darren/cifar-10-py-colmajor/data_batch_6')
z2['data'] = np.single(z2['data'])
z2['data'] /= z2['data'].max()
x = z2['data'] - mean_img_orig
cifar_test_imgs = np.ascontiguousarray(np.single(x.reshape((3, IM_SZ, IM_SZ, CIFAR_FILE_SZ))).transpose((3,0,1,2))[:,np.newaxis])

labels_test_cifar = np.asarray(z2['labels'])
l = np.zeros((CIFAR_FILE_SZ, 10),dtype='uint8')
l[np.arange(CIFAR_FILE_SZ),np.asarray(z2['labels']).astype(int)] = 1
Y_test_cifar = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

def load_cifar(batch, N_CTT, testing=False):
	cifar_batch = batch % (N_IMGS_CIFAR / BATCH_SZ)
	
	if testing:
		assert batch < N_BATCHES_TEST
		cifar_inputs = np.tile(cifar_test_imgs[cifar_batch*BATCH_SZ:(cifar_batch+1)*BATCH_SZ], (1,N_CTT,1,1,1)).reshape((BATCH_SZ,N_CTT*3, IM_SZ, IM_SZ))
		cifar_target = Y_test_cifar[cifar_batch*BATCH_SZ:(cifar_batch+1)*BATCH_SZ]
	else:
		cifar_inputs = np.tile(cifar_imgs[cifar_batch*BATCH_SZ:(cifar_batch+1)*BATCH_SZ], (1,N_CTT,1,1,1)).reshape((BATCH_SZ,N_CTT*3, IM_SZ, IM_SZ))
		cifar_target = Y_cifar[cifar_batch*BATCH_SZ:(cifar_batch+1)*BATCH_SZ]
	
	return cifar_target, cifar_inputs

#############################
# movies
N_MOVIES = 42
MOVIE_FILE_SZ = 2500
MOVIE_FILE_SZ_COMB = 2500*2 # half objs in each file due to memory constraints of rendering
N_FILES_TEST_MOVIE = 2
N_MOVIE_BATCHES = 6
EPOCH_LEN = 16 # length of movie
N_CAT_MOVIE = 10
N_OBJ_MOVIE = 122

N_TEST = MOVIE_FILE_SZ*N_FILES_TEST_MOVIE
N_BATCHES_TEST_MOVIE = N_TEST / BATCH_SZ

z = loadmat('/home/darren/new_movies2/0.mat')
z2 = loadmat('/home/darren/new_movies2/1.mat')

inds = np.arange(N_TEST)

movie_test_imgs = np.concatenate((np.single(z['imgs']), np.single(z2['imgs'])), axis=0)
movie_test_imgs /= movie_test_imgs.max()
movie_test_imgs = movie_test_imgs.reshape((N_TEST, EPOCH_LEN, 3, IM_SZ, IM_SZ))
movie_test_imgs -= mean_img

movie_test_objs = np.concatenate((z['obj_list'].squeeze(), z2['obj_list'].squeeze()))

movie_test_objs = np.ascontiguousarray(movie_test_objs[inds])
movie_test_imgs = np.ascontiguousarray(movie_test_imgs[inds])

l = np.zeros((N_TEST, N_CAT_MOVIE),dtype='uint8')
l[np.arange(N_TEST), syn_cats[movie_test_objs]] = 1
Y_test_movie_cat = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

l = np.zeros((N_TEST, N_OBJ_MOVIE),dtype='uint8')
l[np.arange(N_TEST), movie_test_objs] = 1
Y_test_movie_obj = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

movie_objs = []
movie_imgs = []
Y_movie_cat = []
Y_movie_obj = []

def load_movies(batch, N_CTT, DIFF=False, testing=False):
	global movie_imgs, movie_objs, Y_movie_cat, Y_movie_obj

	movie_batch = batch % (MOVIE_FILE_SZ_COMB / BATCH_SZ)
	
	movie_inputs = np.zeros((BATCH_SZ, N_CTT*3, IM_SZ, IM_SZ), dtype='single')
	frame_target = np.zeros((BATCH_SZ, 3*IM_SZ*IM_SZ, 1), dtype='single')

	obj_target = np.zeros((BATCH_SZ, IM_SZ, 1), dtype='single')
	
	if testing:
		obj_target = Y_test_movie_obj[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		cat_target = Y_test_movie_cat[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		
		objs = movie_test_objs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		cats = syn_cats[movie_test_objs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]]
		
		movie_cur_imgs = movie_test_imgs
	else:
		
		if movie_batch == 0:
			movie_file = ((batch*BATCH_SZ)/MOVIE_FILE_SZ_COMB) % (N_MOVIES/2 - N_FILES_TEST_MOVIE)
			
			z = loadmat('/home/darren/new_movies2/' + str(movie_file*2 + N_FILES_TEST_MOVIE) + '.mat')
			z2 = loadmat('/home/darren/new_movies2/' + str(movie_file*2 + N_FILES_TEST_MOVIE + 1) + '.mat')

			inds = np.arange(N_TEST)

			movie_imgs = np.concatenate((np.single(z['imgs']), np.single(z2['imgs'])), axis=0)
			movie_imgs /= movie_imgs.max()
			movie_imgs = movie_imgs.reshape((N_TEST, EPOCH_LEN, 3, IM_SZ, IM_SZ))
			movie_imgs -= mean_img

			movie_objs = np.concatenate((z['obj_list'].squeeze(), z2['obj_list'].squeeze()))

			movie_objs = np.ascontiguousarray(movie_objs[inds])
			movie_imgs = np.ascontiguousarray(movie_imgs[inds])

			l = np.zeros((N_TEST, N_CAT_MOVIE),dtype='uint8')
			l[np.arange(N_TEST), syn_cats[movie_objs]] = 1
			Y_movie_cat = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories

			l = np.zeros((N_TEST, N_OBJ_MOVIE),dtype='uint8')
			l[np.arange(N_TEST), movie_objs] = 1
			Y_movie_obj = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories
			
		#############
		
		obj_target = Y_movie_obj[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		cat_target = Y_movie_cat[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		
		objs = movie_objs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]
		cats = syn_cats[movie_objs[movie_batch*BATCH_SZ:(movie_batch+1)*BATCH_SZ]]
		
		movie_cur_imgs = movie_imgs
		
	
	for movie in range(BATCH_SZ):
		movie_ind = movie_batch*BATCH_SZ + movie
		movie_frame = np.random.randint(EPOCH_LEN - N_CTT - N_FUTURE + 1) + N_CTT
	
		movie_inputs[movie] = movie_cur_imgs[movie_ind][movie_frame-N_CTT:movie_frame].reshape((1,N_CTT*3, IM_SZ,IM_SZ))

		if DIFF:
			frame_target[movie] = (movie_cur_imgs[movie_ind][movie_frame-1+N_FUTURE] - movie_cur_imgs[movie_ind][movie_frame-1])[np.newaxis].reshape((3*IM_SZ*IM_SZ, 1))
			frame_target[:,0] = .0001
		else:
			frame_target[movie] = movie_cur_imgs[movie_ind][movie_frame-1+N_FUTURE][np.newaxis].reshape((3*IM_SZ*IM_SZ, 1))
	
	return objs,cats, np.ascontiguousarray(cat_target), np.ascontiguousarray(obj_target), np.ascontiguousarray(movie_inputs), np.ascontiguousarray(frame_target)
	

#####################################################################
# imgnet
N_IMGNET_FILES = 118
IMGNET_FILE_SZ = 10000

z = loadmat('/home/darren/imgnet32/data_batch_1')
imgnet_test_imgs = np.single(z['data'])
imgnet_test_imgs /= imgnet_test_imgs.max()
imgnet_test_imgs = imgnet_test_imgs.reshape((IMGNET_FILE_SZ, 1, 3, IM_SZ, IM_SZ))
imgnet_test_imgs -= mean_img

labels_imgnet = z['labels'].squeeze()
l = np.zeros((IMGNET_FILE_SZ, 999),dtype='uint8')
l[np.arange(IMGNET_FILE_SZ), labels_imgnet.astype(int)] = 1
Y_test_imgnet = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories
assert N_BATCHES_TEST*BATCH_SZ <= IMGNET_FILE_SZ

imgnet_imgs = []
Y_imgnet = []

def load_imgnet(batch, N_CTT, testing=False):
	global imgnet_imgs
	global Y_imgnet
	
	imgnet_batch = batch % (IMGNET_FILE_SZ / BATCH_SZ)
	
	if testing:
		assert batch < N_BATCHES_TEST
		imgnet_target = Y_test_imgnet[imgnet_batch*BATCH_SZ:(imgnet_batch+1)*BATCH_SZ]
		imgnet_inputs = np.tile(imgnet_test_imgs[imgnet_batch*BATCH_SZ:(imgnet_batch+1)*BATCH_SZ], (1,N_CTT,1,1,1)).reshape((BATCH_SZ,N_CTT*3, IM_SZ, IM_SZ))
	else:
		
		if imgnet_batch == 0:
			imgnet_file = ((batch*BATCH_SZ)/IMGNET_FILE_SZ) % (N_IMGNET_FILES-1)
			
			z = loadmat('/home/darren/imgnet32/data_batch_' + str(imgnet_file+2))
			imgnet_imgs = np.single(z['data'])
			imgnet_imgs /= imgnet_imgs.max()
			imgnet_imgs = imgnet_imgs.reshape((IMGNET_FILE_SZ, 1, 3, IM_SZ, IM_SZ))
			imgnet_imgs -= mean_img
			
			labels_imgnet = z['labels'].squeeze()
			l = np.zeros((IMGNET_FILE_SZ, 999),dtype='uint8')
			l[np.arange(IMGNET_FILE_SZ), labels_imgnet.astype(int)] = 1
			Y_imgnet = np.ascontiguousarray(np.single(l)[:,:,np.newaxis]) # imgs by categories
		
		imgnet_target = Y_imgnet[imgnet_batch*BATCH_SZ:(imgnet_batch+1)*BATCH_SZ]
		imgnet_inputs = np.tile(imgnet_imgs[imgnet_batch*BATCH_SZ:(imgnet_batch+1)*BATCH_SZ], (1,N_CTT,1,1,1)).reshape((BATCH_SZ,N_CTT*3, IM_SZ, IM_SZ))
		
	return imgnet_target, np.ascontiguousarray(imgnet_inputs)

