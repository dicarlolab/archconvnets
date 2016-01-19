import numpy as np
import time
import scipy.optimize
from scipy.stats import pearsonr
from ntm_core import *

w1 = np.single(np.random.random(123))
w2 = np.single(np.random.random(123))

W1_no_mean = w1 - np.mean(w1)
W2_no_mean = w2 - np.mean(w2)

B = (W1_no_mean * W2_no_mean).sum()

C = (W1_no_mean**2).sum()
D = (W2_no_mean**2).sum()

g2 = (W1_no_mean - (B/D)*W2_no_mean)/np.sqrt(C*D)
	

###########

W1 = init_buffer(w1)
W2 = init_buffer(w2)

LAYER_OUTPUT = init_buffer(pearsonr(w1, w2)[0])
DERIV_ABOVE = init_buffer(np.ones((1,1), dtype='single'))

z = return_buffer(pearson_dinput((W1,W2), LAYER_OUTPUT, DERIV_ABOVE, additional_args=[1])).squeeze()

print np.isclose(z, g2).sum()/np.single(np.prod(g2.shape))

print pearsonr(z,g2)

z = return_buffer(pearson((W1,W2)))

print z, return_buffer(LAYER_OUTPUT), pearsonr(w1, w2)[0]
