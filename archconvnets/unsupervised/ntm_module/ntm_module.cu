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
#include "gradient_functions/focus_key_dbeta_out.c"
#include "gradient_functions/focus_key_dkeys.c"
#include "gradient_functions/sigmoid_dlayer_in.c"
#include "gradient_functions/relu_dlayer_in.c"
#include "gradient_functions/linear_F_dF.c"
#include "gradient_functions/linear_F_dx.c"
#include "gradient_functions/interpolate_do_prev.c"
#include "gradient_functions/interpolate_do_content.c"
#include "gradient_functions/interpolate_dinterp_gate_out.c"
#include "gradient_functions/shift_w_dshift_out.c"
#include "gradient_functions/shift_w_dw_interp.c"
#include "gradient_functions/point_wise_add.c"
#include "gradient_functions/sq_points_dinput.c"
#include "gradient_functions/add_mem.cu"
#include "gradient_functions/add_mem_dgw.cu"
#include "gradient_functions/add_mem_dadd_out.cu"
#include "gradient_functions/point_wise_mult_bcast2.c"

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
	{"focus_key_dbeta_out", focus_key_dbeta_out, METH_VARARGS},
	{"focus_key_dkeys", focus_key_dkeys, METH_VARARGS},
	{"sigmoid_dlayer_in", sigmoid_dlayer_in, METH_VARARGS},
	{"relu_dlayer_in", relu_dlayer_in, METH_VARARGS},
	{"linear_F_dF", linear_F_dF, METH_VARARGS},
	{"linear_F_dx", linear_F_dx, METH_VARARGS},
	{"interpolate_do_prev", interpolate_do_prev, METH_VARARGS},
	{"interpolate_do_content", interpolate_do_content, METH_VARARGS},
	{"interpolate_dinterp_gate_out", interpolate_dinterp_gate_out, METH_VARARGS},
	{"shift_w_dshift_out", shift_w_dshift_out, METH_VARARGS},
	{"shift_w_dw_interp", shift_w_dw_interp, METH_VARARGS},
	{"point_wise_add", point_wise_add, METH_VARARGS},
	{"sq_points_dinput", sq_points_dinput, METH_VARARGS},
	{"add_mem", add_mem, METH_VARARGS},
	{"add_mem_dgw", add_mem_dgw, METH_VARARGS},
	{"add_mem_dadd_out", add_mem_dadd_out, METH_VARARGS},
	{"point_wise_mult_bcast2", point_wise_mult_bcast2, METH_VARARGS},
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
