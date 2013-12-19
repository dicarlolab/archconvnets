# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

#tolist, inds
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
import imagenet
import imagenet.dldatasets
import random
import pickle as pk
import time

# <codecell>

batch_size = 128
img_sz = 138
imgs_mean = np.zeros((3*img_sz*img_sz))

n_batches = 9375
dir_name = '/batch_storage2/batch128_img138_full/'

# <codecell>

dataset = imagenet.dldatasets.ChallengeSynsets2013() # PixelHardSynsets2013ChallengeTop40Screenset()
metacol = 'synset'
preproc = {'crop_rand': 0, 'resize_to': (img_sz, img_sz), 'dtype': 'float32', 'mode': 'RGB', 'seed': 666,
           'normalize': False, 'mask': None, 'crop': None, 'crop_rand_size': 0}
imgs = dataset.get_images(preproc);

# <codecell>

random.seed(666)
inds = np.arange(imgs.shape[0])
random.shuffle(inds)

# <codecell>

labels_unique = np.unique(dataset.meta[metacol])
label_inds = np.zeros((len(dataset.meta)), dtype='int')
for label in range(len(labels_unique)):
    label_inds[dataset.meta[metacol] == labels_unique[label]] = label

# <codecell>

t_start = time.time()
batches_computed = 0
for batch in np.arange(0, 6):#n_batches): #range(n_batches):
    print batch, time.time() - t_start
    t_start = time.time()
    img_offset = batch*batch_size
    img_inds = inds[img_offset:img_offset+batch_size]
    imgs_k = copy.deepcopy(imgs[img_inds])
    imgs_k = imgs_k.transpose([3, 1, 2, 0])
    imgs_k = np.reshape(imgs_k, [3*img_sz*img_sz, batch_size])
    
    #print label_inds[img_inds]
    y = {'batch_label': 'batch_' + str(batch), 'labels': label_inds[img_inds], 'data': np.uint8(np.round(255*imgs_k)), 'filenames': np.asarray(dataset.meta[img_inds]['filename']).tolist()}
    file = open(dir_name + 'data_batch_' + str(batch),'w')
    pk.dump(y, file)
    file.close()
    batches_computed += 1
    imgs_mean += np.mean(np.uint8(np.round(255*imgs_k)), axis=1)
imgs_mean = imgs_mean / batches_computed #n_batches

# <codecell>

y = {'num_cases_per_batch': batch_size, 'label_names': labels_unique,
     'num_vis': 3*img_sz*img_sz, 'data_mean': imgs_mean}
file = open(dir_name + 'batches.meta', 'w')
pk.dump(y, file)
file.close()

# <codecell>


