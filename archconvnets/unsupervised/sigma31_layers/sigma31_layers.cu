#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define IND_DTYPE unsigned long long

#define O3_IND(A,B,C,D)((D) + (C)*max_output_sz3 + (B)*max_output_sz3_max_output_sz3 + (A)*max_output_sz3_max_output_sz3_n3)
#define O2_IND(A,B,C,D)((D) + (C)*max_output_sz2 + (B)*max_output_sz2_max_output_sz2 + (A)*max_output_sz2_max_output_sz2_n2)
#define O1_IND(A,B,C,D)((D) + (C)*max_output_sz1 + (B)*max_output_sz1_max_output_sz1 + (A)*max_output_sz1_max_output_sz1_n1)

#define S1_IND(A,B,C,D,E)((E) + (D)*s1 + (C)*s1_s1 + (B)*s1_s1_n1 + (A)*s1_s1_n1_3)
#define S2_IND(A,B,C,D,E)((E) + (D)*s2 + (C)*s2_s2 + (B)*s2_s2_n1 + (A)*s2_s2_n1_n2)
#define S3_IND(A,B,C,D,E)((E) + (D)*s3 + (C)*s3_s3 + (B)*s3_s3_n2 + (A)*s3_s3_n2_n3)
#define SL_IND(A,B,C,D)((D) + (C)*max_output_sz3 + (B)*max_output_sz3_max_output_sz3 + (A)*max_output_sz3_max_output_sz3_n3)

#define F1_IND(A,B,C,D)(D + (s1)*C + (s1*s1)*B + (s1*s1*n0)*A)
#define F2_IND(A,B,C,D)(D + (s2)*C + (s2*s2)*B + (s2*s2*n1)*A)
#define F3_IND(A,B,C,D)(D + (s3)*C + (s3*s3)*B + (s3*s3*n2)*A)
#define FL_IND(A,B,C,D)(D + (max_output_sz3)*C + (max_output_sz3*max_output_sz3)*B + (max_output_sz3*max_output_sz3*n3)*A)

#define I_IND(A,B,C,D)((D) + (C)*img_sz + (B)*img_sz_img_sz + (A)*img_sz_img_sz_3)


#define CHECK_CUDA_ERR {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return NULL;}}


#define DATA_TYPE_SZ sizeof(float)

#define N_OUTPUTS 10
#define N_SIGMAS 10
#define N_LAYERS 4
#define N_GPUS 4

float * sum_res_c[N_GPUS][N_OUTPUTS];
int deriv_layer_ind_res[N_GPUS][N_OUTPUTS];

int N_C, n1, n0=3, s1, n2, s2, n3, s3, max_output_sz3;

// GPU pointers, one for each GPU
float *F1s_c[N_GPUS], *F2s_c[N_GPUS], *F3s_c[N_GPUS], *FLs_c[N_GPUS];
float *sigma31s_c[N_GPUS][N_SIGMAS]; // second dimension is the layer, generally not all GPUs will have all sigmas.
float *sigma11s_c[N_GPUS];
IND_DTYPE sigma11_len[N_GPUS], *inds_c[N_GPUS], *offsets_c[N_GPUS];
IND_DTYPE n_inds[N_GPUS], n_inds_FL321[N_GPUS];
int N_Cs[N_GPUS];
float *FL321s_c[N_GPUS];

float *F_sum_c[N_GPUS][5], *F_partial_c[N_GPUS][5];
int dims_F_sum[N_GPUS][5][4];

int n1s[N_GPUS][N_SIGMAS], n0s[N_GPUS][N_SIGMAS], s1s[N_GPUS][N_SIGMAS], n2s[N_GPUS][N_SIGMAS], s2s[N_GPUS][N_SIGMAS], n3s[N_GPUS][N_SIGMAS], s3s[N_GPUS][N_SIGMAS], max_output_sz3s[N_GPUS][N_SIGMAS];

int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s[N_GPUS][N_SIGMAS], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s[N_GPUS][N_SIGMAS], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s[N_GPUS][N_SIGMAS],
	max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s[N_GPUS][N_SIGMAS], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s[N_GPUS][N_SIGMAS], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s[N_GPUS][N_SIGMAS],
	max_output_sz3_max_output_sz3_s3_s3_n3_s2s[N_GPUS][N_SIGMAS], max_output_sz3_max_output_sz3_s3_s3_n3s[N_GPUS][N_SIGMAS], max_output_sz3_max_output_sz3_s3_s3s[N_GPUS][N_SIGMAS], 
	max_output_sz3_max_output_sz3_s3s[N_GPUS][N_SIGMAS], max_output_sz3_max_output_sz3s[N_GPUS][N_SIGMAS], z2b[N_GPUS][N_SIGMAS];

#include "set_sigma_buffer.c"
#include "set_sigma11_buffer.c"
#include "set_FL321_buffer.c"
#include "einsum_deriv_gpu.cu"
#include "einsum_return.cu"
#include "set_filter_buffers.cu"

#include "compute_sigma31_full_gpu.cu"
#include "compute_patch_inds.cu"
#include "compute_patch_inds_addresses.cu"
#include "compute_sigma11_gpu.cu"
#include "compute_sigma11.cu"
#include "max_pool_locs.cu"
#include "compute_F_prod_inds.cu"
#include "compute_F_layer_sum_inds.cu"
#include "compute_F_layer_sum_deriv_inds.cu"
#include "compute_F_layer_sum_deriv_inds_gpu.cu"
#include "compute_F_layer_sum_deriv_inds_gpu_return.cu"
#include "compute_sigma11_lin_gpu.cu"
#include "pred_deriv_gpu.cu"
#include "set_img_from_patches.cu"
#include "bp_patch_sigma31.cu"
#include "bp_patch_sigma31_sup.cu"
#include "bp_patch_sigma31_uns.cu"

static PyMethodDef _sigma31_layers[] = {
	{"compute_sigma31_full_gpu", compute_sigma31_full_gpu, METH_VARARGS},
	{"compute_patch_inds", compute_patch_inds, METH_VARARGS},
	{"compute_patch_inds_addresses", compute_patch_inds_addresses, METH_VARARGS},
	{"compute_F_prod_inds", compute_F_prod_inds, METH_VARARGS},
	{"compute_F_layer_sum_inds", compute_F_layer_sum_inds, METH_VARARGS},
	{"compute_F_layer_sum_deriv_inds", compute_F_layer_sum_deriv_inds, METH_VARARGS},
	{"compute_F_layer_sum_deriv_inds_gpu", compute_F_layer_sum_deriv_inds_gpu, METH_VARARGS},
	{"compute_F_layer_sum_deriv_inds_gpu_return", compute_F_layer_sum_deriv_inds_gpu_return, METH_VARARGS},
	{"compute_sigma11", compute_sigma11, METH_VARARGS},
	{"compute_sigma11_gpu", compute_sigma11_gpu, METH_VARARGS},
	{"max_pool_locs", max_pool_locs, METH_VARARGS},
	{"einsum_deriv_gpu", einsum_deriv_gpu, METH_VARARGS},
	{"set_sigma_buffer", set_sigma_buffer, METH_VARARGS},
	{"set_sigma11_buffer", set_sigma11_buffer, METH_VARARGS},
	{"set_FL321_buffer", set_FL321_buffer, METH_VARARGS},
	{"set_filter_buffers", set_filter_buffers, METH_VARARGS},
	{"einsum_return", einsum_return, METH_VARARGS},
	{"compute_sigma11_lin_gpu", compute_sigma11_lin_gpu, METH_VARARGS},
	{"pred_deriv_gpu", pred_deriv_gpu, METH_VARARGS},
	{"set_img_from_patches", set_img_from_patches, METH_VARARGS},
	{"bp_patch_sigma31", bp_patch_sigma31, METH_VARARGS},
	{"bp_patch_sigma31_sup", bp_patch_sigma31_sup, METH_VARARGS},
	{"bp_patch_sigma31_uns", bp_patch_sigma31_uns, METH_VARARGS},
	{NULL, NULL}
};

extern "C" void init_sigma31_layers(){
	(void) Py_InitModule("_sigma31_layers", _sigma31_layers);
	import_array();
	
	for(int gpu = 0; gpu < N_GPUS; gpu++){
		for(int l = 0; l < N_SIGMAS; l++){
			sigma31s_c[gpu][l] = 0;
		}
		for(int layer = 0; layer < N_OUTPUTS; layer++){
			sum_res_c[gpu][layer] = 0;
		}
		F1s_c[gpu] = 0;
		F2s_c[gpu] = 0;
		F3s_c[gpu] = 0;
		FLs_c[gpu] = 0;
		
		sigma11s_c[gpu] = 0;
		FL321s_c[gpu] = 0;
		
		N_Cs[gpu] = 0;
		n_inds[gpu] = 0;
		n_inds_FL321[gpu] = 0;
		
		for(int layer = 0; layer < 5; layer++){
			F_sum_c[gpu][layer] = 0;
			F_partial_c[gpu][layer] = 0;
		}
	}
	return;
} 
