import numpy as np
import time
import scipy.optimize

f_sz = 5
im_sz = 32
I = np.random.random(im_sz**2)
f = np.random.random((f_sz,f_sz))
#F = np.zeros((im_sz**2, im_sz, im_sz)) #np.random.random((im_sz**2, im_sz, im_sz))
F = np.random.random((im_sz**2, im_sz, im_sz)) * 0#1e-15
Fn = np.zeros((im_sz**2, im_sz, im_sz)) #np.random.random((im_sz**2, im_sz, im_sz))

ind = 0
for x_offset in range(im_sz):
	for y_offset in range(im_sz):
		sz1, sz2 = F[ind, x_offset:x_offset+f_sz][:, y_offset:y_offset+f_sz].shape
		
		F[ind, x_offset:x_offset+f_sz][:, y_offset:y_offset+f_sz] = f[:sz1][:, :sz2]
		Fn[ind, x_offset:x_offset+f_sz][:, y_offset:y_offset+f_sz] = f[:sz1][:, :sz2]
		ind += 1

F = F.reshape((im_sz**2, im_sz**2))
Fn = Fn.reshape((im_sz**2, im_sz**2))

O = np.dot(F,I)
On = np.dot(Fn,I)

Ih = np.dot(np.linalg.inv(F), On)
Ihn = np.dot(np.linalg.inv(Fn), On)

print np.linalg.det(F)
print np.isclose(I, Ih).sum()/np.single(np.prod(I.shape))
print np.isclose(I, Ihn).sum()/np.single(np.prod(I.shape))

def f(y):
	weights_shape = F.shape; Fl = F.ravel(); Fl[i_ind] = y
	Fl = F.reshape(weights_shape)
	
	#return np.dot(np.linalg.inv(Fl), O).sum()
	return np.linalg.inv(Fl).sum()

def g(y):
	weights_shape = F.shape; Fl = F.ravel(); Fl[i_ind] = y
	Fl = F.reshape(weights_shape)
	
	dF = np.zeros_like(Fl).ravel()
	dF[i_ind] = 1
	dF = dF.reshape(weights_shape)
	
	Fl_inv = np.linalg.inv(Fl)
	
	g = -np.dot(np.dot(Fl_inv, dF), Fl_inv)
	
	return g.ravel().sum()

ref = F
np.random.seed(np.int64(time.time()))
eps = np.sqrt(np.finfo(np.float).eps)*1e3

N_SAMPLES = 25
ratios = np.zeros(N_SAMPLES)
t_start = time.time()
for sample in range(N_SAMPLES):
	i_ind = np.random.randint(np.prod(ref.shape))
	y = ref.ravel()[i_ind]
	gt = g(y); gtx = scipy.optimize.approx_fprime(np.ones(1)*y, f, eps)
	
	if gtx == 0:
		ratios[sample] = 1
	else:
		ratios[sample] = gtx/gt
	print gt, gtx, ratios[sample]
	
print ratios.mean(), ratios.std(), time.time() - t_start
