#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "includes.h"
#include "set_buffer.c"
#include "free_buffer.c"
#include "return_buffer.c"
#include "sync.c"
#include "return_buffer_sz.c"
#include "copy_buffer.c"
#include "zero_buffer.c"
#include "gradient_functions/dot.cu"
#include "gradient_functions/linear_F_dF.c"
#include "gradient_functions/linear_F_dx.c"
#include "gradient_functions/sum_points.c"
#include "gradient_functions/sum_points_dinput.c"
#include "gradient_functions/point_wise_add.c"
#include "gradient_functions/add_points_dinput.c"
#include "gradient_functions/cosine_sim_dkeys.c"
#include "gradient_functions/cosine_sim_dmem.c"
#include "gradient_functions/cosine_sim.c"
#include "gradient_functions/focus_key_dkeys.c"
#include "gradient_functions/focus_key_dbeta_out.c"
#include "gradient_functions/focus_key.c"
#include "gradient_functions/sigmoid.c"
#include "gradient_functions/sigmoid_dlayer_in.c"
#include "gradient_functions/sharpen.c"
#include "gradient_functions/sharpen_dgamma.c"
#include "gradient_functions/sharpen_dw.c"
#include "gradient_functions/relu.c"
#include "gradient_functions/relu_dlayer_in.c"
#include "gradient_functions/shift_w.c"
#include "gradient_functions/shift_w_dshift_out.c"
#include "gradient_functions/shift_w_dw_interp.c"
#include "gradient_functions/interpolate.c"
#include "gradient_functions/interpolate_do_prev.c"
#include "gradient_functions/interpolate_do_content.c"
#include "gradient_functions/interpolate_dinterp_gate_out.c"
#include "gradient_functions/softmax.c"
#include "gradient_functions/softmax_dlayer_in.c"
#include "gradient_functions/sq_points.c"
#include "gradient_functions/sq_points_dinput.c"
#include "gradient_functions/dotT.c"
#include "gradient_functions/dotT_da.c"
#include "gradient_functions/dotT_db.c"
#include "gradient_functions/mult_points.c"
#include "gradient_functions/mult_points_dinput.c"
#include "gradient_functions/point_wise_div_sqrt.c"
#include "gradient_functions/conv.c"
#include "gradient_functions/conv_ddata.c"
#include "gradient_functions/conv_dfilter.c"
#include "gradient_functions/max_pool.c"
#include "gradient_functions/max_pool_dinput.c"

static PyMethodDef _ntm_module3[] = {
	{"sync", sync, METH_VARARGS},
	{"set_buffer", set_buffer, METH_VARARGS},
	{"copy_buffer", copy_buffer, METH_VARARGS},
	{"free_buffer", free_buffer, METH_VARARGS},
	{"return_buffer", return_buffer, METH_VARARGS},	
	{"return_buffer_sz", return_buffer_sz, METH_VARARGS},
	{"zero_buffer", zero_buffer, METH_VARARGS},
	{"linear_F_dF", linear_F_dF, METH_VARARGS},
	{"linear_F_dx", linear_F_dx, METH_VARARGS},
	{"dot", dot, METH_VARARGS},
	{"sum_points", sum_points, METH_VARARGS},
	{"sum_points_dinput", sum_points_dinput, METH_VARARGS},
	{"point_wise_add", point_wise_add, METH_VARARGS},
	{"add_points_dinput", add_points_dinput, METH_VARARGS},
	{"cosine_sim_dkeys", cosine_sim_dkeys, METH_VARARGS},
	{"cosine_sim_dmem", cosine_sim_dmem, METH_VARARGS},
	{"cosine_sim", cosine_sim, METH_VARARGS},
	{"focus_key_dkeys", focus_key_dkeys, METH_VARARGS},
	{"focus_key_dbeta_out", focus_key_dbeta_out, METH_VARARGS},
	{"focus_key", focus_key, METH_VARARGS},
	{"sigmoid", sigmoid, METH_VARARGS},
	{"sigmoid_dlayer_in", sigmoid_dlayer_in, METH_VARARGS},
	{"sharpen", sharpen, METH_VARARGS},
	{"sharpen_dw", sharpen_dw, METH_VARARGS},
	{"sharpen_dgamma", sharpen_dgamma, METH_VARARGS},
	{"relu", relu, METH_VARARGS},
	{"relu_dlayer_in", relu_dlayer_in, METH_VARARGS},
	{"shift_w", shift_w, METH_VARARGS},
	{"shift_w_dshift_out", shift_w_dshift_out, METH_VARARGS},
	{"shift_w_dw_interp", shift_w_dw_interp, METH_VARARGS},
	{"interpolate", interpolate, METH_VARARGS},
	{"interpolate_do_prev", interpolate_do_prev, METH_VARARGS},
	{"interpolate_do_content", interpolate_do_content, METH_VARARGS},
	{"interpolate_dinterp_gate_out", interpolate_dinterp_gate_out, METH_VARARGS},
	{"softmax", softmax, METH_VARARGS},
	{"softmax_dlayer_in", softmax_dlayer_in, METH_VARARGS},
	{"sq_points", sq_points, METH_VARARGS},
	{"sq_points_dinput", sq_points_dinput, METH_VARARGS},
	{"dotT", dotT, METH_VARARGS},
	{"dotT_da", dotT_da, METH_VARARGS},
	{"dotT_db", dotT_db, METH_VARARGS},
	{"mult_points", mult_points, METH_VARARGS},
	{"mult_points_dinput", mult_points_dinput, METH_VARARGS},
	{"point_wise_div_sqrt", point_wise_div_sqrt, METH_VARARGS},
	{"conv", conv, METH_VARARGS},
	{"conv_ddata", conv_ddata, METH_VARARGS},
	{"conv_dfilter", conv_dfilter, METH_VARARGS},
	{"max_pool", max_pool, METH_VARARGS},
	{"max_pool_dinput", max_pool_dinput, METH_VARARGS},
	{NULL, NULL}
};

extern "C" void init_ntm_module3(){
	(void) Py_InitModule("_ntm_module3", _ntm_module3);
	import_array();
	
	cudnnStatus_t status;
	status = cudnnCreate(&handle);  ERR_CHECK_R
	
	status = cudnnCreatePoolingDescriptor(&poolingDesc);  ERR_CHECK_R
	
	status = cudnnSetPoolingDescriptor(poolingDesc, CUDNN_POOLING_MAX, POOL_WINDOW_SZ, POOL_WINDOW_SZ, POOL_STRIDE, POOL_STRIDE); ERR_CHECK_R
	
	/////////////////////////////////////////////////////////
	for(int gpu_ind = 0; gpu_ind < N_GPUS; gpu_ind++){
		for(int buffer_ind = 0; buffer_ind < N_BUFFERS; buffer_ind++){
			GPU_BUFFER = NULL;
			BUFFER_SZ = 0;
			
			//---------------------------------------
			// Create general Descriptors
			//---------------------------------------
			status = cudnnCreateTensor4dDescriptor(&srcDesc[gpu_ind][buffer_ind]);  ERR_CHECK_R
			status = cudnnCreateTensor4dDescriptor(&gradDesc_data[gpu_ind][buffer_ind]);  ERR_CHECK_R
			status = cudnnCreateTensor4dDescriptor(&destDesc[gpu_ind][buffer_ind]);  ERR_CHECK_R
			status = cudnnCreateFilterDescriptor(&filterDesc[gpu_ind][buffer_ind]);  ERR_CHECK_R
			status = cudnnCreateFilterDescriptor(&gradDesc_filter[gpu_ind][buffer_ind]);  ERR_CHECK_R
			status = cudnnCreateConvolutionDescriptor(&convDesc[gpu_ind][buffer_ind]);  ERR_CHECK_R
			status = cudnnCreateTensor4dDescriptor(&srcDiffDesc[gpu_ind][buffer_ind]);  ERR_CHECK_R
			status = cudnnCreateTensor4dDescriptor(&destDiffDesc[gpu_ind][buffer_ind]);  ERR_CHECK_R
		}
	}
    
	return;
} 
