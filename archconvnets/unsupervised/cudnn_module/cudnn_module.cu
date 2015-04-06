#include "includes.h"

#define CHECK_CUDA_ERR {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return NULL;}}

#define CHECK_CUDA_ERR_R {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return;}}

#include "conv.c"
#include "conv_buffers.c"

#include "conv_dfilter.c"
#include "conv_dfilter_buffers.c"

#include "conv_ddata.c"
#include "conv_ddata_buffers.c"

#include "max_pool_cudnn.c"
#include "max_pool_cudnn_buffers.c"

#include "max_pool_back_cudnn.c"
#include "max_pool_back_cudnn_buffers.c"

#include "set_buffer.c"
#include "set_2d_buffer.c"

#include "return_buffer.c"
#include "return_2d_buffer.c"

#include "pred_buffer.cu"
#include "max_pred_buffer.cu"

static PyMethodDef _cudnn_module[] = {
	{"conv", conv, METH_VARARGS},
	{"conv_buffers", conv_buffers, METH_VARARGS},
	
	{"conv_dfilter", conv_dfilter, METH_VARARGS},
	{"conv_dfilter_buffers", conv_dfilter_buffers, METH_VARARGS},
	
	{"conv_ddata", conv_ddata, METH_VARARGS},
    {"conv_ddata_buffers", conv_ddata_buffers, METH_VARARGS},
    
	{"max_pool_cudnn", max_pool_cudnn, METH_VARARGS},
	{"max_pool_cudnn_buffers", max_pool_cudnn_buffers, METH_VARARGS},
	
	{"max_pool_back_cudnn", max_pool_back_cudnn, METH_VARARGS},
	{"max_pool_back_cudnn_buffers", max_pool_back_cudnn_buffers, METH_VARARGS},
	
	{"set_buffer", set_buffer, METH_VARARGS},
	{"set_2d_buffer", set_2d_buffer, METH_VARARGS},
	
	{"return_buffer", return_buffer, METH_VARARGS},
	{"return_2d_buffer", return_2d_buffer, METH_VARARGS},
	
	{"pred_buffer", pred_buffer, METH_VARARGS},
	{"max_pred_buffer", max_pred_buffer, METH_VARARGS},
	{NULL, NULL}
};

extern "C" void init_cudnn_module(){
	(void) Py_InitModule("_cudnn_module", _cudnn_module);
	import_array();
	
	cudnnStatus_t status;
	status = cudnnCreate(&handle);  ERR_CHECK_R
	
	cudaError_t err;
	for(int gpu = 0; gpu < N_GPUS; gpu++){
		cudaSetDevice(gpu); CHECK_CUDA_ERR_R
		cudaStreamCreate(&streams[gpu]); CHECK_CUDA_ERR_R
		for(int alt_stream = 0; alt_stream < N_ALT_STREAMS; alt_stream++){
			cudaStreamCreate(&alt_streams[gpu][alt_stream]); CHECK_CUDA_ERR_R
		}
	}
	
	//---------------------------------------
	// Create general Descriptors
	//---------------------------------------
	status = cudnnCreateTensor4dDescriptor(&srcDesc);  ERR_CHECK_R
	status = cudnnCreateTensor4dDescriptor(&gradDesc_data);  ERR_CHECK_R
	status = cudnnCreateTensor4dDescriptor(&destDesc);  ERR_CHECK_R
	status = cudnnCreateFilterDescriptor(&filterDesc);  ERR_CHECK_R
	status = cudnnCreateFilterDescriptor(&gradDesc_filter);  ERR_CHECK_R
	status = cudnnCreateConvolutionDescriptor(&convDesc);  ERR_CHECK_R
	
	status = cudnnCreatePoolingDescriptor(&poolingDesc);  ERR_CHECK_R
	status = cudnnCreateTensor4dDescriptor(&srcDiffDesc);  ERR_CHECK_R
	status = cudnnCreateTensor4dDescriptor(&destDiffDesc);  ERR_CHECK_R
	
	status = cudnnSetPoolingDescriptor(poolingDesc, CUDNN_POOLING_MAX, POOL_WINDOW_SZ, POOL_WINDOW_SZ, POOL_STRIDE, POOL_STRIDE); ERR_CHECK_R
	
    /////////////////////////////////////////////////////////
    for(int gpu = 0; gpu < N_GPUS; gpu++){
		for(int buffer = 0; buffer < N_BUFFERS; buffer++){
			data_buffers[gpu][buffer] = 0;
			data_2d_buffers[gpu][buffer] = 0;
		}
	}
    
	return;
} 
