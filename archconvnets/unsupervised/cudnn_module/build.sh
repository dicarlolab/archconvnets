#nvcc -c cudnn_module.cu -I/usr/include/python2.7  -I/usr/include/numpy --compiler-options '-fPIC -std=gnu99 '
nvcc -c cudnn_module.cu -I/usr/include/python2.7  -I/usr/include/numpy --compiler-options '-fPIC' -O3 -arch sm_20
gcc -o _cudnn_module.so  cudnn_module.o -lpython2.7 -shared -lcudnn -L/home/darren/cudnn-6.5-linux-R1 -lcuda -L/usr/local/cuda/lib64 -lcudart
