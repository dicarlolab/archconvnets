#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "includes.h"
#include "set_buffer.c"
#include "free_buffer.c"
#include "return_buffer.c"
#include "sync.c"
#include "gradient_functions/dot.cu"
#include "gradient_functions/cosine_sim_expand_dkeys_cpu.c"
#include "gradient_functions/cosine_sim_expand_dmem_cpu.c"
#include "gradient_functions/cosine_sim_expand_dkeys.c"
#include "gradient_functions/cosine_sim_expand_dmem.c"
#include "gradient_functions/softmax_dlayer_in_cpu.c"
#include "gradient_functions/softmax_dlayer_in.c"
#include "gradient_functions/sharpen_dw_cpu.c"
#include "gradient_functions/sharpen_dw.c"
#include "gradient_functions/sharpen_dgamma_cpu.c"
#include "gradient_functions/sharpen_dgamma.c"

static PyMethodDef _ntm_module[] = {
	{"sync", sync, METH_VARARGS},
	{"set_buffer", set_buffer, METH_VARARGS},
	{"free_buffer", free_buffer, METH_VARARGS},
	{"dot", dot, METH_VARARGS},
	{"return_buffer", return_buffer, METH_VARARGS},	
	{"cosine_sim_expand_dkeys_cpu", cosine_sim_expand_dkeys_cpu, METH_VARARGS},
	{"cosine_sim_expand_dmem_cpu", cosine_sim_expand_dmem_cpu, METH_VARARGS},
	{"cosine_sim_expand_dkeys", cosine_sim_expand_dkeys, METH_VARARGS},
	{"cosine_sim_expand_dmem", cosine_sim_expand_dmem, METH_VARARGS},
	{"softmax_dlayer_in_cpu", softmax_dlayer_in_cpu, METH_VARARGS},
	{"softmax_dlayer_in", softmax_dlayer_in, METH_VARARGS},
	{"sharpen_dw_cpu", sharpen_dw_cpu, METH_VARARGS},
	{"sharpen_dw", sharpen_dw, METH_VARARGS},
	{"sharpen_dgamma_cpu", sharpen_dgamma_cpu, METH_VARARGS},
	{"sharpen_dgamma", sharpen_dgamma, METH_VARARGS},
	{NULL, NULL}
};

extern "C" void init_ntm_module(){
	(void) Py_InitModule("_ntm_module", _ntm_module);
	import_array();
	
	/////////////////////////////////////////////////////////
	for(int gpu_ind = 0; gpu_ind < N_GPUS; gpu_ind++){
		for(int buffer_ind = 0; buffer_ind < N_BUFFERS; buffer_ind++){
			GPU_BUFFER = NULL;
			BUFFER_SZ = 0;
		}
	}
    
	return;
} 
