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
	printf("line: %i\n", __LINE__);}
#define ERR_CHECK {if (status != CUDNN_STATUS_SUCCESS){STATUSES;return NULL;}}
#define ERR_CHECK_R {if (status != CUDNN_STATUS_SUCCESS){STATUSES;return;}}
#define MALLOC_ERR_CHECK {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__); return NULL;}}

#define DATA_TYPE_SZ sizeof(float)
cudnnHandle_t handle;
cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
