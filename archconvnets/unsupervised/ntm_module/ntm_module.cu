#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "includes.h"
#include "set_buffer.c"
#include "free_buffer.c"
#include "dot.cu"
#include "return_buffer.c"
#include "sync.c"
#include "cosine_sim_expand_dkeys_cpu.c"
#include "cosine_sim_expand_dkeys.c"

static PyMethodDef _ntm_module[] = {
	{"sync", sync, METH_VARARGS},
	{"set_buffer", set_buffer, METH_VARARGS},
	{"free_buffer", free_buffer, METH_VARARGS},
	{"dot", dot, METH_VARARGS},
	{"return_buffer", return_buffer, METH_VARARGS},	
	{"cosine_sim_expand_dkeys_cpu", cosine_sim_expand_dkeys_cpu, METH_VARARGS},
	{"cosine_sim_expand_dkeys", cosine_sim_expand_dkeys, METH_VARARGS},
	{NULL, NULL}
};

extern "C" void init_ntm_module(){
	(void) Py_InitModule("_ntm_module", _ntm_module);
	import_array();
	
	cudaError_t err;
	
	/////////////////////////////////////////////////////////
    for(int gpu_ind = 0; gpu_ind < N_GPUS; gpu_ind++){
		cudaSetDevice(gpu_ind); CHECK_CUDA_ERR_R
		for(int buffer_ind = 0; buffer_ind < N_BUFFERS; buffer_ind++){
			GPU_BUFFER = NULL;
			BUFFER_SZ = 0;
		}
	}
    
	return;
} 
