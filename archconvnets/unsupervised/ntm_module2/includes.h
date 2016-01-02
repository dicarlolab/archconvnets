#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG 1

#define CHECK_CUDA_ERR {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return NULL;}}

#define CHECK_CUDA_ERR_R {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return;}}

#define MALLOC_ERR_CHECK {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__); return NULL;}}

#define DATA_TYPE float
#define MALLOC(A, B) {A = (DATA_TYPE *)malloc(B); if(A == NULL){printf("malloc err line: %i\n",__LINE__);}}

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 65535
#define THREAD_CAPACITY (MAX_BLOCKS*MAX_THREADS_PER_BLOCK)

#define N_BUFFERS 1024
#define N_GPUS 4

#define GPU_BUFFER gpu_buffers[gpu_ind][buffer_ind]
#define BUFFER_SZ buffer_sz[gpu_ind][buffer_ind]

#define OUT_BUFFER_SZ buffer_sz[gpu_ind][out_buffer_ind]
#define GPU_BUFFER_OUT gpu_buffers[gpu_ind][out_buffer_ind]

// BUFFER_SZ: size of buffer in bytes

float *gpu_buffers[N_GPUS][N_BUFFERS];
unsigned long buffer_sz[N_GPUS][N_BUFFERS];

