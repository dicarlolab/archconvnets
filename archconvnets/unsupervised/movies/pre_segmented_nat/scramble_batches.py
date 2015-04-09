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

z = np.load('/export/storage/UCF101_80ex_per_cat/data_batch_' + str(1))
t_start = time.time()
for batch in range(2,100-5):
	if batch % 10 == 0:
		print batch, time.time() - t_start
		t_start = time.time()
	y = np.load('/export/storage/UCF101_80ex_per_cat/data_batch_' + str(batch))
	z['cat_inds'] = np.concatenate((z['cat_inds'], y['cat_inds']))
	z['file_inds'] = np.concatenate((z['file_inds'], y['file_inds']))
	z['data'] = np.concatenate((z['data'], y['data']), axis=1)

inds = range(len(z['cat_inds']))
random.seed(666)
random.shuffle(inds)

z['cat_inds'] = z['cat_inds'][inds]
z['file_inds'] = z['file_inds'][inds]
z['data'] = z['data'][:, inds]

for batch in range(1,100-5):
	t = batch - 1
	file = open('/export/storage/UCF101_80ex_per_cat_scrambled_5heldout/data_batch_' + str(batch),'w')
	pk.dump({'data':z['data'][:,t*10000:(t+1)*10000], 'mean': z['mean'], 'file_inds': z['file_inds'][t*10000:(t+1)*10000], 'cat_inds':z['cat_inds'][t*10000:(t+1)*10000]}, file)
	file.close()
