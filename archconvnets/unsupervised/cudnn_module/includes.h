#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#define MALLOC_ERR_CHECK {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__); return NULL;}}

#define DATA_TYPE_SZ sizeof(float)

#define POOL_WINDOW_SZ 3
#define POOL_STRIDE 2

cudnnHandle_t handle;
cudnnDataType_t dataType = CUDNN_DATA_FLOAT;


#define N_BUFFERS 100
#define N_GPUS 4

cudnnTensor4dDescriptor_t desc_buffers[N_GPUS][N_BUFFERS];
cudnnFilterDescriptor_t desc_filters[N_GPUS][N_BUFFERS];
int data_dims[4][N_GPUS][N_BUFFERS];
float *data_buffers[N_GPUS][N_BUFFERS];
int filter_flags[N_GPUS][N_BUFFERS];

cudaStream_t streams[N_GPUS];
