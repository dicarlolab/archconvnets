archconvnets
==============

Architecturally optimized convolutional neural networks trained with regularized backpropagation


install
==============
git clone this repository and add the path to the PYTHON_PATH variable

follow the install instructions for all requirements listed in requirements.txt
(including the requirements in those requirements files)

you have to download the cifar-10 dataset, and then set the environment variable 
CIFAR10_PATH to its location (untarred) to run tests properly:
```
cd ~/.skdata
wget http://www.cs.toronto.edu/~kriz/cifar-10-py-colmajor.tar.gz
export CIFAR10_PATH=~/.skdata/cifar-10-py-colmajor
```

If you're on a machine other than honeybadger (or one that is similarly configured)
modify archconvnets/convnet/build.sh to match your machine's setup (cuda, python, and numpy locations must be specified)


Compile
```
sh build.sh
```

running tests
=================
you must be in the archconvnets/convnet directory to run tests:
```
nosetests tests
```
=======
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
