import numpy as np
import time
from ntm_core import *

N_IMGS = 2

deriv_above = np.single(np.random.random((2,3, N_IMGS, 4, 5)))
layer_out = np.single(np.random.random((N_IMGS, 4,5)))
a = np.single(np.random.random((N_IMGS, 4,5)))
b = np.single(np.random.random((4,5)))

DERIV_ABOVE = init_buffer(deriv_above)
LAYER_OUT = init_buffer(layer_out)
A = init_buffer(a)
B = init_buffer(b)

deriv = deriv_above.sum(2)

deriv_test = return_buffer(add_points_batch_dinputB((A,B), LAYER_OUT, DERIV_ABOVE))

print np.isclose(deriv, deriv_test).sum()/np.single(np.prod(deriv.shape))


