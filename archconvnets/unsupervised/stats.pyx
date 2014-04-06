import numpy as np
cimport numpy as np

def threestats(np.ndarray[np.float64_t, ndim=2] X):

    cdef int s0 = X.shape[0]
    cdef int s1 = X.shape[1]
    
    s = X.std(1)
    s = np.outer(s, np.outer(s,s)).reshape((s0, s0, s0))

    X = X - X.mean(1)[:, np.newaxis]

    cdef Py_ssize_t i, j, k    
    cdef np.ndarray B = np.zeros([s0, s0, s0], np.float64)
    
    for i in range(s0):
        for j in range(i, s0):
            for k in range(j, s0):
                 B[i, j, k] = (X[i] * X[j] * X[k]).mean()
                 B[i, k, j] = B[i, j, k]
                 B[j, i, k] = B[i, j, k]
                 B[j, k, i] = B[i, j, k]
                 B[k, i, j] = B[i, j, k]
                 B[k, j, i] = B[i, j, k]
                 
    return B / s
    

def threestats_flat(np.ndarray[np.float64_t, ndim=2] X):

    cdef int s0 = X.shape[0]
    cdef int s1 = X.shape[1]
    

    cdef Py_ssize_t i, j, k, ind 
    
    cdef int s = (s0**3 + 3*s0**2 + 2*s0)/6
    
    ind = 0
    cdef np.ndarray S = np.zeros([s], np.float64)
    for i in range(s0):
        for j in range(i, s0):
            for k in range(j, s0):
                S[ind] = X[i].std() * X[j].std() * X[k].std()
                ind = ind + 1

    X = X - X.mean(1)[:, np.newaxis]
    
    ind = 0
    cdef np.ndarray B = np.zeros([s], np.float64)
    for i in range(s0):
        for j in range(i, s0):
            for k in range(j, s0):
                 B[ind] = (X[i] * X[j] * X[k]).mean()
                 ind = ind + 1
                 
    return B / S