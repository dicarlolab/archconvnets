nvcc -c ntm_module3.cu -I/usr/include/python2.7  -I/usr/include/numpy --compiler-options '-fPIC' -O3 -arch sm_20 --use_fast_math
gcc -o _ntm_module3.so  ntm_module3.o -lpython2.7 -shared -lcudnn -L/home/darren/cudnn-6.5-linux-R1 -lcuda -L/usr/local/cuda/lib64 -lcudart -lcublas -O3
