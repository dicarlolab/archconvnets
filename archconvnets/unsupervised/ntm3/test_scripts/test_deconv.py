import numpy as np

f_sz = 5
im_sz = 32
I = np.random.random(im_sz**2)
f = np.random.random((f_sz,f_sz))
F = np.zeros((im_sz**2, im_sz, im_sz))

ind = 0
for x_offset in range(im_sz):
	for y_offset in range(im_sz):
		sz1, sz2 = F[ind, x_offset:x_offset+f_sz][:, y_offset:y_offset+f_sz].shape
		F[ind, x_offset:x_offset+f_sz][:, y_offset:y_offset+f_sz] = f[:sz1][:, :sz2]
		ind += 1

F = F.reshape((im_sz**2, im_sz**2))

O = np.dot(F,I)

Ih = np.dot(np.linalg.inv(F), O)


