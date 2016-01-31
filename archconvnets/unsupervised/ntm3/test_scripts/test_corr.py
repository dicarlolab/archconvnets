import numpy as np
import time
from ntm_core import *
from scipy.stats import pearsonr

a = np.single(np.random.random((1200,5))) - .5
b = np.single(np.random.random((1200,5))) - .5

A = init_buffer(a)
B = init_buffer(b)

z = return_buffer(pearson((A,B)))
print z, pearsonr(a.ravel(),b.ravel())