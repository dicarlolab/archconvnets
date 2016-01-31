import numpy as np
import time
from ntm_core import *

a = np.single(np.random.random((12,5))) - .5
deriv_above = np.single(np.random.random((2,3,12,5)))

A = init_buffer(a)
DERIV_ABOVE = init_buffer(deriv_above)

a2 = copy.deepcopy(a)

a2[a < 0] = 0

LAYER_OUT = init_buffer(a2)

DERIV = relu_dlayer_in((A,), LAYER_OUT, DERIV_ABOVE)
deriv = return_buffer(DERIV)

deriv_test = copy.deepcopy(deriv_above)
for i in range(2):
	for j in range(3):
		for k in range(12):
			for l in range(5):
				if a[k,l] < 0:
					deriv_test[i,j,k,l] = 0