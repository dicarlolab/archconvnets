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

p = '/home/darren/archconvnets/archconvnets/unsupervised/movies/pre_segmented/mp4s/'
files = [ f for f in listdir(p) if isfile(join(p,f)) ]

n_files = len(files)
n_frames = 130

imgs = np.zeros((3*32*32, n_files*n_frames),dtype='uint8')
labels = np.zeros(n_files*n_frames, dtype='int')

for cat in range(n_files):
	print '----------------', cat
	os.system('ffmpeg -i mp4s/' + files[cat] + ' -r 30 -f image2 /tmp/image-%3d.png')
	
	labels[cat*n_frames:(cat+1)*n_frames] = cat
	for frame in range(1,n_frames+1):
		frame_string = '%03i' % frame
		imgs[:,cat*n_frames + frame - 1] = np.asarray(PIL.Image.open('/tmp/image-' + frame_string + '.png').resize((32,32))).transpose((2,0,1)).ravel()

file = open('/home/darren/archconvnets/archconvnets/unsupervised/movies/pre_segmented/movies_batch','w')
pk.dump({'data':imgs, 'mean': imgs.mean(1)[:,np.newaxis], 'labels': labels}, file)
file.close()


