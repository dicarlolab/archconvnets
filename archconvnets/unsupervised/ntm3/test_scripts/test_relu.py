import numpy as np
import time
from ntm_core import *

x = 50
y = 100

a = np.single(np.random.random((x,y))) - .5
deriv_above = np.single(np.random.random((2,3,x,y)))

A = init_buffer(a)
DERIV_ABOVE = init_buffer(deriv_above)

a2 = copy.deepcopy(a)

a2[a < 0] = 0

LAYER_OUT = init_buffer(a2)
LAYER_OUT_GPU = relu((A,))

print np.isclose(a2, return_buffer(LAYER_OUT_GPU)).sum()/np.single(np.prod(a2.shape))

DERIV = relu_dlayer_in((A,), LAYER_OUT, DERIV_ABOVE)
deriv = return_buffer(DERIV)

deriv_test = copy.deepcopy(deriv_above)
for i in range(2):
	for j in range(3):
		for k in range(x):
			for l in range(y):
				if a[k,l] < 0:
					deriv_test[i,j,k,l] = 0
					
print np.isclose(deriv, deriv_test).sum()/np.single(np.prod(deriv_test.shape))