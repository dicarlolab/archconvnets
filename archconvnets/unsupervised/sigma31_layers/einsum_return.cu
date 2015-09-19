static PyObject *einsum_return(PyObject *self, PyObject *args){
	cudaError_t err;
	PyArrayObject *sum_res_in;
	
	float *sum_res;
	
	int gpu_ind, output_ind;
	int dims[7];
	
	if (!PyArg_ParseTuple(args, "ii", &output_ind, &gpu_ind)) 
		return NULL;
	int g = gpu_ind;
	int deriv_layer_ind = deriv_layer_ind_res[g][output_ind];
	
	if(output_ind < 0 || output_ind > N_OUTPUTS){
		printf("invalid output_ind %i\n", output_ind);
		return NULL;
	}
	
	if(deriv_layer_ind < 1 || deriv_layer_ind > N_LAYERS){
		printf("invalid deriv_layer_ind %i\n", deriv_layer_ind);
		return NULL;
	}
	
	if(g < 0 || g > N_GPUS){
		printf("invalid gpu index %i\n", g);
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	////////////////////////////////////////////////////////////////////////// which indices do we unravel across threads?
	int output_sz, n_dims;

	dims[0] = N_C;
	if(deriv_layer_ind == 1){ // F1 deriv
		output_sz = N_C * n1 * n0 * s1 * s1;
		dims[1] = n1;
		dims[2] = n0;
		dims[3] = s1;
		dims[4] = s1;
		n_dims = 5;
	}else if(deriv_layer_ind == 2){ // F2 deriv
		output_sz = N_C * n2 * n1 * s2 * s2;
		dims[1] = n2;
		dims[2] = n1;
		dims[3] = s2;
		dims[4] = s2;
		n_dims = 5;
	}else if(deriv_layer_ind == 3){ // F3 deriv
		output_sz = N_C * n3 * n2 * s3 * s3;
		dims[1] = n3;
		dims[2] = n2;
		dims[3] = s3;
		dims[4] = s3;
		n_dims = 5;
	}else if(deriv_layer_ind == 4){ // FL deriv
		output_sz = N_C * n3 * max_output_sz3 * max_output_sz3;
		dims[1] = n3;
		dims[2] = max_output_sz3;
		dims[3] = max_output_sz3;
		n_dims = 4;
	}
	
	///////////////////////////////// allocate output mem
	sum_res_in = (PyArrayObject *) PyArray_FromDims(n_dims, dims, NPY_FLOAT);
	sum_res = (float *) sum_res_in -> data;
	
	/////////////////////////////////// cuda mem
	if(sum_res_c[gpu_ind][output_ind] == 0){
		printf("output buffer is empty. call einsum_deriv_gpu() before trying to get return values\n");
		return NULL;
	}

	cudaThreadSynchronize();
	
	CHECK_CUDA_ERR
	
	cudaMemcpy(sum_res, sum_res_c[gpu_ind][output_ind], output_sz * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  CHECK_CUDA_ERR
	
	cudaFree(sum_res_c[gpu_ind][output_ind]);
	sum_res_c[gpu_ind][output_ind] = 0;
	
	return PyArray_Return(sum_res_in);
}

