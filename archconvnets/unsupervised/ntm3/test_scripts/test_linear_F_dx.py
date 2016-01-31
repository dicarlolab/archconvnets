import numpy as np
import time
from ntm_core import *

f = np.single(np.random.random((12,5)))
x = np.single(np.random.random((5,4)))
deriv_above = np.single(np.random.random((2,3,12,4)))

F = init_buffer(f)
X = init_buffer(x)
DERIV_ABOVE = init_buffer(deriv_above)

LAYER_OUT = init_buffer(np.dot(f,x))

DERIV = linear_F_dx((F,X), LAYER_OUT, DERIV_ABOVE, additional_args=[False,False])
deriv = return_buffer(DERIV)

deriv_test = np.einsum(deriv_above, [0,1,2,3], f, [2,4], [0,1,4,3])

print np.isclose(deriv_test, deriv).sum()/np.single(np.prod(deriv.shape))
