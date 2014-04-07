import numpy as np
import math
cimport numpy as np

def threestats_flat(np.ndarray[np.float32_t, ndim=2] X):
	cdef int in_dims = X.shape[0]
	cdef int N = X.shape[1]
	cdef Py_ssize_t i, j, k, ind 
	cdef int n_triplets = math.factorial(in_dims)/(math.factorial(in_dims-3)*6)
	
	target = np.random.random((n_triplets)) # for debugging -- target stats to match
	
	x_no_mean = X - X.mean(1)[:, np.newaxis]
	x_std = X.std(1)
	x_no_mean_div_std_N = x_no_mean/(N*x_std[:,np.newaxis])
	
	cdef np.ndarray w = np.zeros([in_dims,in_dims,N], np.float32)
	cdef np.ndarray w_std = np.zeros([in_dims,in_dims], np.float32)
	for m in range(in_dims):
		for n in range(m+1, in_dims):
			w[m,n] = x_no_mean[m]*x_no_mean[n]
			w_std[m,n] = x_std[m] * x_std[n]
	w_no_mean = w - w.mean(2)[:,:,np.newaxis]
	
	ind = 0
	cdef np.ndarray numer = np.zeros([n_triplets], np.float32)
	cdef np.ndarray denom = np.zeros([n_triplets], np.float32)
	for i in range(in_dims):
		for m in range(i+1, in_dims):
			for n in range(m+1, in_dims):
				numer[ind] = (x_no_mean[i] * w[m,n]).sum() # will change back to mean once I can reproduce our prior results
				denom[ind] = x_std[i] * w_std[m,n]
				ind = ind + 1
	stat_mat = numer / denom
	loss = np.sum(np.abs(target - stat_mat))
	
	numer = numer / (denom ** 2)
	denom = 1 / denom # = denom / denom2
	
	###################### up until this point the code is equivilant in computation time to stats.threestats_flat()
	ind -= 1 # debug... this should be determined from indices: r,m,n
	cdef np.ndarray grad = np.zeros((in_dims, N), np.float32)
	for r in range(in_dims):
		for m in range(in_dims):
			if m != r:
				for n in range(m+1, in_dims):
					if n != r:
						sign = 1
						if stat_mat[ind] < target[ind]:
							sign = -1
						grad_denom = w_std[m,n]*x_no_mean_div_std_N[r]
						grad[r] += sign*(w_no_mean[m,n]*denom[ind] - numer[ind]*grad_denom)
	return stat_mat
