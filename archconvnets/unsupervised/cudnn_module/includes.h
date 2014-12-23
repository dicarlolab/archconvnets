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
cudnnHandle_t handle;
cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

/*int n_img_buffers = 0, n_filter_buffers = 0, n_conv_buffers = 0;
cudnnTensor4dDescriptor_t * srcDesc_buffers = NULL;
cudnnTensor4dDescriptor_t * destDesc_buffers = NULL;
cudnnFilterDescriptor_t * filterDesc_buffers = NULL;
cudnnConvolutionDescriptor_t * convDesc_buffers = NULL;

float **srcData_buffers = NULL, **filterData_buffers = NULL, **destData_buffers = NULL;

int *n_channels_imgs_buffers = NULL, *n_channels_filters_buffers = NULL, *filter_sz_buffers = NULL, *n_filters_buffers = NULL, *img_sz_buffers = NULL, *n_imgs_buffers = NULL, *dims_buffers = NULL, *conv_filter_ind = NULL, *conv_img_ind = NULL;*/

#define N_BUFFERS 512

int n_img_buffers = N_BUFFERS, n_filter_buffers = N_BUFFERS, n_conv_buffers = N_BUFFERS;
cudnnTensor4dDescriptor_t srcDesc_buffers[N_BUFFERS];
cudnnTensor4dDescriptor_t destDesc_buffers[N_BUFFERS];
cudnnFilterDescriptor_t filterDesc_buffers[N_BUFFERS];
cudnnConvolutionDescriptor_t convDesc_buffers[N_BUFFERS];

float *srcData_buffers[N_BUFFERS], *filterData_buffers[N_BUFFERS], *destData_buffers[N_BUFFERS];

int n_channels_imgs_buffers[N_BUFFERS], n_channels_filters_buffers[N_BUFFERS], filter_sz_buffers[N_BUFFERS], n_filters_buffers[N_BUFFERS], img_sz_buffers[N_BUFFERS], n_imgs_buffers[N_BUFFERS], dims_buffers[N_BUFFERS];
int conv_filter_ind[N_BUFFERS], conv_img_ind[N_BUFFERS];

