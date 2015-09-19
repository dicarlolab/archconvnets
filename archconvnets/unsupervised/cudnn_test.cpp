// export LD_LIBRARY_PATH=/home/darren/cudnn-6.5-linux-R1:$LD_LIBRARY_PATH
// nvcc cudnn_test.cpp -lcudnn -L/home/darren/cudnn-6.5-linux-R1 -o t;./t
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "/home/darren/cudnn-6.5-linux-R1/cudnn.h"

#define ERR_CHECK {if (status != CUDNN_STATUS_SUCCESS){if(status == CUDNN_STATUS_NOT_INITIALIZED) printf("CUDNN_STATUS_NOT_INITIALIZED\n"); \
	if(status == CUDNN_STATUS_ALLOC_FAILED) printf("CUDNN_STATUS_ALLOC_FAILED\n");\
	if(status == CUDNN_STATUS_BAD_PARAM) printf("CUDNN_STATUS_BAD_PARAM\n");\
	if(status == CUDNN_STATUS_ARCH_MISMATCH) printf("CUDNN_STATUS_ARCH_MISMATCH\n");\
	if(status == CUDNN_STATUS_MAPPING_ERROR) printf("CUDNN_STATUS_MAPPING_ERROR\n");\
	if(status == CUDNN_STATUS_EXECUTION_FAILED) printf("CUDNN_STATUS_EXECUTION_FAILED\n");\
	if(status == CUDNN_STATUS_INTERNAL_ERROR) printf("CUDNN_STATUS_INTERNAL_ERROR\n");\
	if(status == CUDNN_STATUS_NOT_SUPPORTED) printf("CUDNN_STATUS_NOT_SUPPORTED\n");\
	if(status == CUDNN_STATUS_LICENSE_ERROR) printf("CUDNN_STATUS_LICENSE_ERROR\n");\
	printf("line: %i\n", __LINE__);return -1;}}
#define MALLOC_ERR_CHECK {if (err != cudaSuccess){printf("malloc err line: %i\n",__LINE__); return -1;}}

int main(){
cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
int data_type_sz = sizeof(float);
cudnnHandle_t handle;
int n_imgs = 32;
int n_imgs_out;
int n_channels = 16;
int img_sz = 64;
int n_filters = 16;
int n_filters_out;
int filter_sz = 5;
int conv_out_sz_x;
int conv_out_sz_y;
cudaError_t err;

float *srcData;
float *filterData;
float *destData;
float *local_buffer = NULL;

size_t workspace_size;
cudnnStatus_t status;

cudnnTensor4dDescriptor_t srcDesc;
cudnnFilterDescriptor_t filterDesc;
cudnnConvolutionDescriptor_t convDesc;
cudnnTensor4dDescriptor_t destDesc;

//---------------------------------------
// Create CudNN
//---------------------------------------
status = cudnnCreate(&handle);   ERR_CHECK

//---------------------------------------
// Create Descriptors
//---------------------------------------
status = cudnnCreateTensor4dDescriptor(&srcDesc);  ERR_CHECK
status = cudnnCreateTensor4dDescriptor(&destDesc);  ERR_CHECK
status = cudnnCreateFilterDescriptor(&filterDesc);  ERR_CHECK
status = cudnnCreateConvolutionDescriptor(&convDesc);  ERR_CHECK

//---------------------------------------
// Set decriptors
//---------------------------------------
status = cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
status = cudnnSetFilterDescriptor(filterDesc, dataType, n_filters, n_channels, filter_sz, filter_sz);  ERR_CHECK
status = cudnnSetConvolutionDescriptor(convDesc, srcDesc, filterDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION);  ERR_CHECK

//---------------------------------------
// Query output layout
//---------------------------------------
status = cudnnGetOutputTensor4dDim(convDesc, CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK
printf("%i %i %i %i\n", n_imgs_out, n_filters_out, conv_out_sz_x, conv_out_sz_y);

//--------------------------------------
// Set and allocate output tensor descriptor
//----------------------------------------
status = cudnnSetTensor4dDescriptor(destDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs_out, n_filters_out, conv_out_sz_x, conv_out_sz_x); ERR_CHECK

err = cudaMalloc(&destData, n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x * data_type_sz); MALLOC_ERR_CHECK

//--------------------------------------
// allocate filter, image, alpha, and beta tensors
//----------------------------------------
err = cudaMalloc(&srcData, n_imgs*n_channels*img_sz*img_sz * data_type_sz); MALLOC_ERR_CHECK
err = cudaMalloc(&filterData, n_filters*n_channels*filter_sz*filter_sz * data_type_sz); MALLOC_ERR_CHECK

local_buffer = (float*)malloc(2*n_imgs*n_channels*img_sz*img_sz * data_type_sz);
if(local_buffer == NULL){
	printf("failed to alloc local buffer\n");
	return -1;
}
memset(local_buffer, 1, 2*n_imgs*n_channels*img_sz*img_sz * data_type_sz); 
for(int i = 0; i < 2*n_imgs*n_channels*img_sz*img_sz; i++){
	local_buffer[i] = i/1e8;
}
printf("%f\n", local_buffer[10]);
//--------------------------------------
// set filter and image values
//--------------------------------------
err = cudaMemcpy(srcData, local_buffer, n_imgs*n_channels*img_sz*img_sz * data_type_sz, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
err = cudaMemcpy(filterData, local_buffer, n_filters*n_channels*filter_sz*filter_sz * data_type_sz, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK


//--------------------------------------
// Convolution
//--------------------------------------
printf("start\n");
status = cudnnConvolutionForward(handle, srcDesc, srcData, filterDesc, filterData, convDesc, destDesc, destData, CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK
printf("finish\n");

//--------------------------------------
// Get output data
//------------------------------------------
printf("%f\n", local_buffer[10]);

err = cudaMemcpy(local_buffer, destData, n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x * data_type_sz, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK

float sum = 0;
for(int i = 0; i < n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x; i++){
	sum += local_buffer[i];
}
printf("%f\n", sum);

return 0;
}

