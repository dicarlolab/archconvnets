#!/bin/sh

# Fill in these environment variables.
# I have tested this code with CUDA 4.0, 4.1, and 4.2. 
# Only use Fermi-generation cards. Older cards won't work.

# If you're not sure what these paths should be, 
# you can use the find command to try to locate them.
# For example, NUMPY_INCLUDE_PATH contains the file
# arrayobject.h. So you can search for it like this:
# 
# find /usr -name arrayobject.h
# 
# (it'll almost certainly be under /usr)

# CUDA toolkit installation directory.
#export CUDA_INSTALL_PATH=/cm/shared/apps/cuda55/toolkit/5.5.22
export CUDA_INSTALL_PATH=/om/user/yamins/usr/local/cuda

# CUDA SDK installation directory.
export CUDA_SDK_PATH=/om/user/yamins/src/NVIDIA_GPU_Computing_SDK42

# Python include directory. This should contain the file Python.h, among others.
export PYTHON_INCLUDE_PATH=/om/user/yamins/usr/local/include/python2.7

# Numpy include directory. This should contain the file arrayobject.h, among others.
export NUMPY_INCLUDE_PATH=/om/user/yamins/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy

# ATLAS library directory. This should contain the file libcblas.so, among others.
export ATLAS_LIB_PATH=/cm/shared/openmind/openblas/0.2.9.rc2-singlethread/lib

make $*

