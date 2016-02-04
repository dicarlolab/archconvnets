#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "/home/darren/cudnn-6.5-linux-R1/cudnn.h"

#define STATUSES {if(status == CUDNN_STATUS_NOT_INITIALIZED) printf("CUDNN_STATUS_NOT_INITIALIZED\n"); \
	if(status == CUDNN_STATUS_ALLOC_FAILED) printf("CUDNN_STATUS_ALLOC_FAILED\n");\
	if(status == CUDNN_STATUS_BAD_PARAM) printf("CUDNN_STATUS_BAD_PARAM\n");\
	if(status == CUDNN_STATUS_ARCH_MISMATCH) printf("CUDNN_STATUS_ARCH_MISMATCH\n");\
	if(status == CUDNN_STATUS_MAPPING_ERROR) printf("CUDNN_STATUS_MAPPING_ERROR\n");\
	if(status == CUDNN_STATUS_EXECUTION_FAILED) printf("CUDNN_STATUS_EXECUTION_FAILED\n");\
	if(status == CUDNN_STATUS_INTERNAL_ERROR) printf("CUDNN_STATUS_INTERNAL_ERROR\n");\
	if(status == CUDNN_STATUS_NOT_SUPPORTED) printf("CUDNN_STATUS_NOT_SUPPORTED\n");\
	if(status == CUDNN_STATUS_LICENSE_ERROR) printf("CUDNN_STATUS_LICENSE_ERROR\n");\
	printf("%s line: %i\n", __FILE__, __LINE__);}
#define ERR_CHECK {if (status != CUDNN_STATUS_SUCCESS){STATUSES;return NULL;}}
#define ERR_CHECK_R {if (status != CUDNN_STATUS_SUCCESS){STATUSES;return;}}
#define ERR_CHECK_BLAS {if (err_blas != CUBLAS_STATUS_SUCCESS){printf("blas err. %s line: %i\n",__FILE__,__LINE__); return NULL;}}
#define ERR_CHECK_BLAS_R {if (err_blas != CUBLAS_STATUS_SUCCESS){printf("blas err. %s line: %i\n",__FILE__,__LINE__); return;}}
#define MALLOC_ERR_CHECK {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__); return NULL;}}


//#define DEBUG 1
//#define TIMING_DEBUG 1

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

#define N_BUFFERS 5000
#define N_GPUS 4

#define GPU_BUFFER gpu_buffers[gpu_ind][buffer_ind]
#define BUFFER_SZ buffer_sz[gpu_ind][buffer_ind]

#define OUT_BUFFER_SZ buffer_sz[gpu_ind][out_buffer_ind]
#define GPU_BUFFER_OUT gpu_buffers[gpu_ind][out_buffer_ind]


#define DATA_TYPE_SZ sizeof(float)

float *gpu_buffers[N_GPUS][N_BUFFERS];
unsigned long buffer_sz[N_GPUS][N_BUFFERS];

cudnnTensor4dDescriptor_t srcDesc[N_GPUS][N_BUFFERS];
cudnnFilterDescriptor_t filterDesc[N_GPUS][N_BUFFERS];
cudnnConvolutionDescriptor_t convDesc[N_GPUS][N_BUFFERS];
cudnnTensor4dDescriptor_t destDesc[N_GPUS][N_BUFFERS];
cudnnTensor4dDescriptor_t gradDesc_data[N_GPUS][N_BUFFERS];
cudnnFilterDescriptor_t gradDesc_filter[N_GPUS][N_BUFFERS];
cudnnTensor4dDescriptor_t srcDiffDesc[N_GPUS][N_BUFFERS];
cudnnTensor4dDescriptor_t destDiffDesc[N_GPUS][N_BUFFERS];

cudnnPoolingDescriptor_t poolingDesc;
cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

#define POOL_WINDOW_SZ 3
#define POOL_STRIDE 2

cudnnHandle_t handle[N_GPUS];
cublasHandle_t handle_blas[N_GPUS];

char device_init[N_GPUS];
