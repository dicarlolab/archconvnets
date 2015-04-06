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

# ffmpeg -i seven.mp4 -r 15 -f image2 /tmp/img-%5d.png

n_batches = 9

imgs = np.zeros((3*32*32, n_batches*10000),dtype='uint8')

for batch in range(n_batches):
	imgs[:,batch*10000:(batch+1)*10000] = np.load('/home/darren/archconvnets/archconvnets/unsupervised/movies/nat_even_segment/seven_batch_' + str(batch))['data']

file = open('/home/darren/archconvnets/archconvnets/unsupervised/movies/nat_even_segment/combined_batch', 'w')
pk.dump({'data':imgs, 'mean': imgs.mean(1)[:,np.newaxis]}, file)
file.close()


