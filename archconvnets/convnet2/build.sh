#!/bin/sh
# Copyright 2014 Google Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

# Fill in the below environment variables.
#
# If you're not sure what these paths should be, 
# you can use the find command to try to locate them.
# For example, NUMPY_INCLUDE_PATH contains the file
# arrayobject.h. So you can search for it like this:
# 
# find /usr -name arrayobject.h
# 
# (it'll almost certainly be under /usr)

# CUDA toolkit installation directory.
#export CUDA_INSTALL_PATH=/usr/local/cuda
export CUDA_INSTALL_PATH=/cm/shared/apps/cuda55/toolkit/5.5.22

# Python include directory. This should contain the file Python.h, among others.
#export PYTHON_INCLUDE_PATH=/usr/include/python2.7
export PYTHON_INCLUDE_PATH=/om/user/yamins/usr/local/include/python2.7

export PYTHON_LIB_PATH=/om/user/yamins/usr/local/lib

# Numpy include directory. This should contain the file arrayobject.h, among others.
#export NUMPY_INCLUDE_PATH=/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
export NUMPY_INCLUDE_PATH=/om/user/yamins/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy

#opencv export
export OPENCV_INCLUDE_PATH=/om/user/yamins/usr/local/include

# ATLAS library directory. This should contain the file libcblas.so, among others.
export ATLAS_LIB_PATH=/om/user/yamins/usr/local/lib/atlas

# You don't have to change these:
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH
export CUDA_SDK_PATH=$CUDA_INSTALL_PATH/samples
export PATH=$PATH:$CUDA_INSTALL_PATH/bin

cd util && make clean && make numpy=1 -j $* && cd ..
cd nvmatrix && make -j $* && cd ..
cd cudaconv3 && make -j $* && cd ..
cd cudaconvnet && make -j $* && cd ../..
cd /om/user/yamins/src/archconvnets/archconvnets/convnet2/make-data/pyext && make -j $* && cd ../..
