#include "einsum_kernel.cu"

static PyObject *einsum_deriv_gpu(PyObject *self, PyObject *args){
	cudaError_t err;
	
	int deriv_layer_ind; // which dimensions to sum over
	int l; // sigma31 buffer ind
	int output_ind; // buffer to store output
	int gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iiii", &deriv_layer_ind, &l, &output_ind, &gpu_ind)) 
		return NULL;
	int g = gpu_ind;
	
	if(l < 0 || l > N_SIGMAS){
		printf("invalid sigma index %i\n", l);
		return NULL;
	}
	
	if(g < 0 || g > N_GPUS){
		printf("invalid gpu index %i\n", g);
		return NULL;
	}
	
	if(output_ind < 0 || output_ind > N_OUTPUTS){
		printf("invalid output_ind %i\n", output_ind);
		return NULL;
	}
	
	if(deriv_layer_ind < 0 || deriv_layer_ind > N_LAYERS){
		printf("invalid deriv_layer_ind %i\n", deriv_layer_ind);
		return NULL;
	}
	
	if(sum_res_c[g][output_ind] != 0){
		printf("output buffer used, call sigma_return first, for gpu %i, sigma_ind %i, output_ind: %i\n", gpu_ind, l, output_ind);
		return NULL;
	}
	
	if(sigma31s_c[g][l] == 0){
		printf("sigma buffer not initialized on gpu %i for layer %i\n", gpu_ind, l);
		return NULL;
	}
	
	if(F1s_c[g] == 0){
		printf("filter buffers not initialized on gpu %i\n", g);
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	
	////////////////////////////////////////////////////////////////////////// which indices do we unravel across threads?
	int output_sz;
	dim3 thread_sz;
	dim3 grid_sz;

	if(deriv_layer_ind == 0){ // prediction (no deriv)
		thread_sz.x = s1*s2*s2*s3;
		//thread_sz.y = n0;
		output_sz = N_C * N_C;
		grid_sz.x = N_C;
		grid_sz.y = N_C;
	}else if(deriv_layer_ind == 1){ // F1 deriv
		thread_sz.x = s2*s2*s3*s3;
		output_sz = N_C * N_C * n1 * n0 * s1 * s1;
		grid_sz.x = N_C * N_C * s1 * s1;
		grid_sz.y = n1;
		grid_sz.z = n0;
	}else if(deriv_layer_ind == 2){ // F2 deriv
		thread_sz.x = s1*s1*s3*s3;
		output_sz = N_C * N_C * n2 * n1 * s2 * s2;
		grid_sz.x = N_C * N_C * s2 * s2;
		grid_sz.y = n2;
		grid_sz.z = n1;
	}else if(deriv_layer_ind == 3){ // F3 deriv
		thread_sz.x = s1*s1*s2;//*s2;
		output_sz = N_C * N_C * n3 * n2 * s3 * s3;
		grid_sz.x = N_C * N_C * s3 * s3;
		grid_sz.y = n3;
		grid_sz.z = n2;
	}else if(deriv_layer_ind == 4){ // FL deriv
		thread_sz.x = s1*s1*s2;//*s2;
		output_sz = N_C * n3 * max_output_sz3 * max_output_sz3;
		grid_sz.x = N_C * max_output_sz3;
		grid_sz.y = max_output_sz3;
		grid_sz.z = n3;
	}
	
	
	deriv_layer_ind_res[g][output_ind] = deriv_layer_ind;
	
	/////////////////////////////////// cuda mem
	
	cudaMalloc((void**) &sum_res_c[g][output_ind], output_sz * DATA_TYPE_SZ); CHECK_CUDA_ERR
	
	
	// indexing products
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l]*n3s[g][l]*s2s[g][l]*s2s[g][l]*n2s[g][l]*s1s[g][l]*s1s[g][l]*n0s[g][l]*n1s[g][l];
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l]*n3s[g][l]*s2s[g][l]*s2s[g][l]*n2s[g][l]*s1s[g][l]*s1s[g][l]*n0s[g][l];
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l]*n3s[g][l]*s2s[g][l]*s2s[g][l]*n2s[g][l]*s1s[g][l]*s1s[g][l];
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l]*n3s[g][l]*s2s[g][l]*s2s[g][l]*n2s[g][l]*s1s[g][l];
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l]*n3s[g][l]*s2s[g][l]*s2s[g][l]*n2s[g][l];
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l]*n3s[g][l]*s2s[g][l]*s2s[g][l];
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l]*n3s[g][l]*s2s[g][l];
	int max_output_sz3_max_output_sz3_s3_s3_n3s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l]*n3s[g][l];
	int max_output_sz3_max_output_sz3_s3_s3s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l]*s3s[g][l];
	int max_output_sz3_max_output_sz3_s3s = max_output_sz3s[g][l]*max_output_sz3s[g][l]*s3s[g][l];
	int max_output_sz3_max_output_sz3s = max_output_sz3s[g][l]*max_output_sz3s[g][l];
	int max_output_sz3s_local = max_output_sz3s[g][l];
	int z2b = 1;
	
	// check which dims should be broadcasted
	if(n1s[g][l] != n1){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s = 0;
	}
	if(n0s[g][l] != n0){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s = 0;
	}
	if(s1s[g][l] != s1){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s = 0;
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s = 0;
	}
	if(n2s[g][l] != n2){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s = 0;
	}
	if(s2s[g][l] != s2){
		max_output_sz3_max_output_sz3_s3_s3_n3s = 0;
		max_output_sz3_max_output_sz3_s3_s3_n3_s2s = 0;
	}
	if(s3s[g][l] != s3){
		max_output_sz3_max_output_sz3s = 0;
		max_output_sz3_max_output_sz3_s3s = 0;
	}
	if(n3s[g][l] != n3){
		max_output_sz3_max_output_sz3_s3_s3s = 0;
	}
	if(max_output_sz3s[g][l] != max_output_sz3){
		max_output_sz3s_local = 0;
		z2b = 0;
	}
	
	//////////////////////////////////////////////////////////////////////////
	
	kernel_deriv <<< grid_sz, thread_sz, DATA_TYPE_SZ >>> (sum_res_c[g][output_ind], sigma31s_c[g][l], F1s_c[g], F2s_c[g], F3s_c[g], FLs_c[g],
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s, max_output_sz3_max_output_sz3_s3_s3_n3_s2s, max_output_sz3_max_output_sz3_s3_s3_n3s, max_output_sz3_max_output_sz3_s3_s3s,
		max_output_sz3_max_output_sz3_s3s, max_output_sz3_max_output_sz3s, z2b, n0, n0s[g][l], n1, n1s[g][l], n2, n2s[g][l], n3, n3s[g][l],
		max_output_sz3, max_output_sz3s_local, s1, s1s[g][l], s2, s2s[g][l], s3, s3s[g][l], N_C, deriv_layer_ind);
	
	
	CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
