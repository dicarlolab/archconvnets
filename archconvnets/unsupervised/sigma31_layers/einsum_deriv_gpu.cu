#include "einsum_kernel.cu"

static PyObject *einsum_deriv_gpu(PyObject *self, PyObject *args){
	cudaError_t err;
	
	int gpu_ind, l, deriv_ind = 0, deriv_flag; // l: layer_ind
	
	if (!PyArg_ParseTuple(args, "iii", &l, &deriv_flag, &gpu_ind)) 
		return NULL;
	
	if(cudaSetDevice(gpu_ind) != cudaSuccess){
		err = cudaGetLastError();
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return NULL;
	}
	
	if(deriv_flag){ //otherwise the prediction for each category is computed (N_C by N_C) output
		deriv_ind = l;
	}
	
	////////////////////////////////////////////////////////////////////////// which indices do we unravel across threads?
	int output_sz;
	dim3 thread_sz;
	dim3 grid_sz;

	if(deriv_ind == 0){ // prediction (no deriv)
		thread_sz.x = s1*s2*s2*s3;
		//thread_sz.y = n0;
		output_sz = N_C * N_C;
		grid_sz.x = N_C;
		grid_sz.y = N_C;
	}else if(deriv_ind == 1){ // F1 deriv
		thread_sz.x = s2*s2*s3*s3;
		output_sz = N_C * N_C * n1 * n0 * s1 * s1;
		grid_sz.x = N_C * N_C * s1 * s1;
		grid_sz.y = n1;
		grid_sz.z = n0;
	}else if(deriv_ind == 2){ // F2 deriv
		thread_sz.x = s1*s1*s3*s3;
		output_sz = N_C * N_C * n2 * n1 * s2 * s2;
		grid_sz.x = N_C * N_C * s2 * s2;
		grid_sz.y = n2;
		grid_sz.z = n1;
	}else if(deriv_ind == 3){ // F3 deriv
		thread_sz.x = s1*s1*s2*s2;
		output_sz = N_C * N_C * n3 * n2 * s3 * s3;
		grid_sz.x = N_C * N_C * s3 * s3;
		grid_sz.y = n3;
		grid_sz.z = n2;
	}else if(deriv_ind == 4){ // FL deriv
		thread_sz.x = s1*s1*s2*s2;
		output_sz = N_C * n3 * max_output_sz3 * max_output_sz3;
		grid_sz.x = N_C * max_output_sz3;
		grid_sz.y = max_output_sz3;
		grid_sz.z = n3;
	}
	
	err = cudaMalloc((void**) &sum_res_c[gpu_ind][l][deriv_flag], output_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	
	kernel_deriv <<< grid_sz, thread_sz, DATA_TYPE_SZ >>> (sum_res_c[gpu_ind][l][deriv_flag], sigma31s_c[gpu_ind][l], F1s_c[gpu_ind], F2s_c[gpu_ind], F3s_c[gpu_ind], FLs_c[gpu_ind], 
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s[l], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s[l],
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s[l], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s[l], max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s[l],
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s[l], max_output_sz3_max_output_sz3_s3_s3_n3_s2s[l], max_output_sz3_max_output_sz3_s3_s3_n3s[l], max_output_sz3_max_output_sz3_s3_s3s[l],
		max_output_sz3_max_output_sz3_s3s[l], max_output_sz3_max_output_sz3s[l], z2b[l], n0, n0s[l], n1, n1s[l], n2, n2s[l], n3, n3s[l],
		max_output_sz3, max_output_sz3s[l], s1, s1s[l], s2, s2s[l], s3, s3s[l], N_C, deriv_ind);
	
	
	Py_INCREF(Py_None);
	return Py_None;
}
