import numpy as np
import time
from ntm_core import *

n_imgs = 2
dim_above = 1

X = random_function((n_imgs,3,4))
F = random_function((5,3))

Fg = init_buffer(F); Xg = init_buffer(X)

deriv_above = random_function((n_imgs,dim_above, 5,4))

DERIV_ABOVE = init_buffer(deriv_above)
LAYER_OUT = linear_F((Fg,Xg))

dF = return_buffer(linear_F_dF((Fg,Xg), LAYER_OUT, DERIV_ABOVE, additional_args=[False,False]))

dF_target = np.zeros((n_imgs, dim_above) + F.shape, dtype='single')

for img in range(n_imgs):
	for batch in range(dim_above):
		dF_target[img,batch] = np.dot(deriv_above[img,batch], X[img].T)


print np.isclose(dF, dF_target.sum(0).squeeze()).sum()/np.single(np.prod(dF.shape))
