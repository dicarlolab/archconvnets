import numpy as np
import time
from ntm_core import *

a = np.single(np.random.random((12,4)))
deriv_above = np.single(np.random.random((2,12,4)))

A = init_buffer(a)
DERIV_ABOVE = init_buffer(deriv_above)

LAYER_OUT = softmax([A])

DERIV = softmax_dlayer_in([A], LAYER_OUT, DERIV_ABOVE)


