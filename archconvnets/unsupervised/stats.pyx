import numpy as np
cimport numpy as np

def threestats(np.ndarray[np.float64_t, ndim=2] X, int slim0=0, int slim1=-1):

    cdef int s0 = X.shape[0]
    cdef int s1 = X.shape[1]

    if slim1 < 0:
        slim1 = s0
    cdef int ssize = slim1 - slim0

    s = X.std(1)[slim0: slim1]
    s = np.outer(s, np.outer(s,s)).reshape((ssize, ssize, ssize))

    X = X - X.mean(1)[:, np.newaxis]

    cdef Py_ssize_t i, j, k

    cdef np.ndarray B = np.zeros([ssize, ssize, ssize], np.float64)

    cdef int ilim, jlim

    for i in range(slim0, slim1):
        ilim = max(slim0, i)
        for j in range(ilim, slim1):
            jlim = max(slim0, j)
            for k in range(jlim, slim1):
                 B[i-slim0, j-slim0, k-slim0] = (X[i] * X[j] * X[k]).mean()
                 B[i-slim0, k-slim0, j-slim0] = B[i-slim0, j-slim0, k-slim0]
                 B[j-slim0, i-slim0, k-slim0] = B[i-slim0, j-slim0, k-slim0]
                 B[j-slim0, k-slim0, i-slim0] = B[i-slim0, j-slim0, k-slim0]
                 B[k-slim0, i-slim0, j-slim0] = B[i-slim0, j-slim0, k-slim0]
                 B[k-slim0, j-slim0, i-slim0] = B[i-slim0, j-slim0, k-slim0]
                 
    return B / s