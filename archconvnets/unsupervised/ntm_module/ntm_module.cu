#include "includes.h"

#define CHECK_CUDA_ERR {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return NULL;}}

#define CHECK_CUDA_ERR_R {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return;}}

#include "set_buffer.c"

static PyMethodDef _ntm_module[] = {
	
	{"set_buffer", set_buffer, METH_VARARGS},
	{NULL, NULL}
};

extern "C" void init_ntm_module(){
	(void) Py_InitModule("_ntm_module", _ntm_module);
	import_array();
	
	cudaError_t err;
	for(int gpu = 0; gpu < N_GPUS; gpu++){
		cudaSetDevice(gpu); CHECK_CUDA_ERR_R
	}
	
	/////////////////////////////////////////////////////////
    	for(int gpu = 0; gpu < N_GPUS; gpu++){
		for(int buffer = 0; buffer < N_BUFFERS; buffer++){
			data_buffers[gpu][buffer] = 0;
		}
	}
    
	return;
} 
