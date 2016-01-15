import time
from ntm_core import *

IMGS = init_buffer(np.asarray(np.random.random((1,4,32,32)),dtype='single'))
F = init_buffer(np.asarray(np.random.random((5,4,3,3)),dtype='single'))

O = conv((F,IMGS))
DIMGS = conv_ddata((F,IMGS), O)
DF = conv_dfilter((F,IMGS), O)

t_start = time.time()
O = conv((F,IMGS))
DIMGS = conv_ddata((F,IMGS), O)
DF = conv_dfilter((F,IMGS), O)
print time.time() - t_start

print 'IMGS', IMGS[1]
print 'F', F[1]
print 'O', O[1]
print 'DIMGS', DIMGS[1]
print 'DF', DF[1]
