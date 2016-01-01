#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "includes.h"
#include "set_buffer.c"
#include "free_buffer.c"
#include "return_buffer.c"
#include "sync.c"
#include "return_buffer_sz.c"
#include "gradient_functions/dot.cu"
#include "gradient_functions/linear_F_dF.c"
#include "gradient_functions/linear_F_dx.c"
#include "gradient_functions/sum_points.c"
#include "gradient_functions/sum_points_dinput.c"
#include "gradient_functions/point_wise_add.c"
#include "gradient_functions/add_points_dinput.c"

static PyMethodDef _ntm_module2[] = {
	{"sync", sync, METH_VARARGS},
	{"set_buffer", set_buffer, METH_VARARGS},
	{"free_buffer", free_buffer, METH_VARARGS},
	{"return_buffer", return_buffer, METH_VARARGS},	
	{"return_buffer_sz", return_buffer_sz, METH_VARARGS},
	{"linear_F_dF", linear_F_dF, METH_VARARGS},
	{"linear_F_dx", linear_F_dx, METH_VARARGS},
	{"dot", dot, METH_VARARGS},
	{"sum_points", sum_points, METH_VARARGS},
	{"sum_points_dinput", sum_points_dinput, METH_VARARGS},
	{"point_wise_add", point_wise_add, METH_VARARGS},
	{"add_points_dinput", add_points_dinput, METH_VARARGS},
	{NULL, NULL}
};

extern "C" void init_ntm_module2(){
	(void) Py_InitModule("_ntm_module2", _ntm_module2);
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
