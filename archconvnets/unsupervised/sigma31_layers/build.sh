nvcc -c sigma31_layers.cu -I/usr/include/python2.7  -I/usr/include/numpy --compiler-options '-fPIC' -O3 -arch sm_20
gcc -flat_namespace -o _sigma31_layers.so  sigma31_layers.o -lpython2.7 -shared -lcuda -L/usr/local/cuda/lib64 -lcudart
