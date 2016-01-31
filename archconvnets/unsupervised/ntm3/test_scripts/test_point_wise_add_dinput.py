import numpy as np
import time
from ntm_core import *

a = np.single(np.random.random((12)))
b = np.single(np.random.random((12)))
deriv_above = np.single(np.random.random((2,3,12)))

A = init_buffer(a)
B = init_buffer(b)
DERIV_ABOVE = init_buffer(deriv_above)

LAYER_OUT = init_buffer(a+b*2.3)

DERIV = add_points_dinput((A,B), LAYER_OUT, DERIV_ABOVE, additional_args=[2.3])
