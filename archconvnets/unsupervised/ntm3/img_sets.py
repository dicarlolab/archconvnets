import numpy as np
from scipy.io import loadmat
from gpu_flag import *
import Image

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
N_BATCHES_TEST_MOVIE = 10
N_MOVIES = 15931
EPOCH_LEN = 11 # length of movie

def load_movies(N_CTT, DIFF=False, testing=False):
	cats = np.zeros(BATCH_SZ)
	objs = np.zeros(BATCH_SZ)
	
	movie_inputs = np.zeros((BATCH_SZ, N_CTT*3, IM_SZ, IM_SZ), dtype='single')
	frame_target = np.zeros((BATCH_SZ, 3*32*32, 1), dtype='single')
	cat_target = np.zeros((BATCH_SZ, 8, 1), dtype='single')
	obj_target = np.zeros((BATCH_SZ, 32, 1), dtype='single')

	for img in range(BATCH_SZ):
		movie_frame = np.random.randint(EPOCH_LEN - N_CTT - N_FUTURE + 1) + N_CTT
		
		if testing == False:
			movie_ind = np.random.randint(N_MOVIES - N_BATCHES_TEST_MOVIE*BATCH_SZ) + N_BATCHES_TEST_MOVIE*BATCH_SZ
		else:
			movie_ind = img + testing*BATCH_SZ
		
		z = loadmat('/home/darren/rotating_objs32_constback_const_movement_25t/imgs' + str(movie_ind)  + '.mat')
		
		cats[img] = z['cat'][0][0]
		objs[img] = z['obj'][0][0]
		
		cat_target[img, cats[img]] = 1
		obj_target[img, objs[img]] = 1
		
		movie_inputs[img] = (z['imgs'][movie_frame-N_CTT:movie_frame] - mean_img).reshape((1,N_CTT*3, IM_SZ, IM_SZ))
		
		#temp = np.asarray(Image.fromarray(np.uint8(255*(z['imgs'][movie_frame-1+N_FUTURE] - mean_img)).reshape((3,32,32)).transpose((1,2,0))).resize((16,16)),dtype='single')/255
		#frame_target[img] = temp.transpose((2,0,1)).reshape((3*16*16,1))
		if DIFF:
			frame_target[img] = (z['imgs'][movie_frame-1+N_FUTURE][np.newaxis] - z['imgs'][movie_frame-1][np.newaxis]).reshape((3*IM_SZ*IM_SZ, 1))
			frame_target[:,0] = .0001
		else:
			frame_target[img] = (z['imgs'][movie_frame-1+N_FUTURE][np.newaxis] - mean_img).reshape((3*IM_SZ*IM_SZ, 1))

	movie_inputs = np.ascontiguousarray(movie_inputs)
	
	return objs,cats, cat_target, obj_target, movie_inputs, frame_target

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
		imgnet_file = ((batch*BATCH_SZ)/IMGNET_FILE_SZ) % (N_IMGNET_FILES-1)
		
		if imgnet_batch == 0:
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

