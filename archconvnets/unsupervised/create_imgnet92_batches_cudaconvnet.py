# python convnet.py --data-path=/export/storage/imgnet92_cudaconvnet --crop=0 --save-path=/home/darren/imgnet92_64 --test-range=1-2 --train-range=3-15 --layer-def=/home/darren/archconvnets/archconvnets/convnet/ut_model_full/layer64.cfg --layer-params=/home/darren/archconvnets/archconvnets/convnet/ut_model_full/layer-params64.cfg --data-provider=general-cropped --test-freq=5 --conserve-mem=1 --gpu=3 --scale-rate=0.01
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import PIL
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
import scipy.io as sio
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import scipy.io as sio
import numpy as np
import tabular as tb
import copy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import random
from scipy.stats import rankdata
import random
import pickle as pk
import time
import matplotlib.image as mpimg
import Image

N_BATCHES = 9374
BATCH_SZ = 256

labels = np.zeros(BATCH_SZ, dtype='int')
imgs_new = np.zeros((3*92*92, BATCH_SZ), dtype='uint8')

write_batch = 1
img_i_global = 0
filenames = [None]*BATCH_SZ
t_start = time.time()
data_mean = np.zeros((3*92*92))
for batch in range(1,N_BATCHES+1):
	#print batch, img_i_global
	z = np.load('/export/batch_storage2/batch128_img138_full/data_batch_' + str(batch))
	imgs = z['data'].reshape((3,138,138,128)).transpose((3,1,2,0))
	
	for img_i in range(128):
		img_local = img_i_global % BATCH_SZ
		imgs_new[:,img_local] = np.asarray(Image.fromarray(imgs[img_i]).resize((92,92))).transpose((2,0,1)).ravel()
		data_mean += imgs_new[:,img_local]
		labels[img_local] = z['labels'][img_i]
		filenames[img_local] = z['filenames'][img_i]
		if img_local == (BATCH_SZ-1):
			print 'writing', write_batch, time.time() - t_start
			t_start = time.time()
			file = open('/export/storage/imgnet92_cudaconvnet/data_batch_' + str(write_batch-1),'w')
			pk.dump({'data':imgs_new, 'mean': imgs_new.mean(1)[:,np.newaxis], 'labels': labels, 'batch_label': str(write_batch)}, file)
			file.close()
			write_batch += 1
		
		img_i_global += 1
data_mean /= img_i_global
z = np.load('/export/batch_storage2/batch128_img138_full/batches.meta')
z['num_cases_per_batch'] = BATCH_SZ
z['num_vis'] = 3*92*92
z['data_mean'] = copy.deepcopy(data_mean)

file = open('/export/storage/imgnet92_cudaconvnet/batches.meta','w')
pk.dump(z, file)
file.close()


