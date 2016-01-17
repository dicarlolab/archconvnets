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
from scipy.io import savemat

# avconv -i test_clip.avi -r 4 -f image2 /tmp/img-%5d.png

N_IMGS = 2784

imgs = np.zeros((N_IMGS, 3,32,32),dtype='single')

for frame in range(N_IMGS):
	frame_string = '%05i' % (1 + frame)
	imgs[frame] = np.asarray(PIL.Image.open('/tmp/img-' + frame_string + '.png').resize((48,32)))[:,7:7+32].transpose((2,0,1))

savemat('/home/darren/test_clip.mat', {'imgs': imgs})
