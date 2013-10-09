archconvnets
==============

Architecturally optimized convolutional neural networks trained with regularized backpropagation. For now, mostly a wrapper around
http://cs.nyu.edu/~wanli/dropc/




installing
==========
pip install git+http://github.com/dicarlolab/archconvnets

install cuda 4.2, drivers and the SDK

call ldconfig at the install location of cuda

install libboost

change build to match your local settings (cuda location, python location, numpy location)

sh build.sh


There is more extensive documentation at
https://code.google.com/p/cuda-convnet/
which forms the convolutional neural network "backend"
