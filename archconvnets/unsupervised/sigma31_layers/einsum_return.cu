static PyObject *einsum_return(PyObject *self, PyObject *args){
	cudaError_t err;
	PyArrayObject *sum_res_in;
	
	float *sum_res;
	
	int gpu_ind, l, deriv_ind = 0, deriv_flag; // l: layer_ind
	int dims[7];
	
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
	int output_sz, n_dims;

	dims[0] = N_C;
	if(deriv_ind == 0){ // prediction (no deriv)
		output_sz = N_C * N_C;
		dims[1] = N_C;
		n_dims = 2;
	}else if(deriv_ind == 1){ // F1 deriv
		output_sz = N_C * N_C * n1 * n0 * s1 * s1;
		dims[1] = N_C;
		dims[2] = n1;
		dims[3] = n0;
		dims[4] = s1;
		dims[5] = s1;
		n_dims = 6;
	}else if(deriv_ind == 2){ // F2 deriv
		output_sz = N_C * N_C * n2 * n1 * s2 * s2;
		dims[1] = N_C;
		dims[2] = n2;
		dims[3] = n1;
		dims[4] = s2;
		dims[5] = s2;
		n_dims = 6;
	}else if(deriv_ind == 3){ // F3 deriv
		output_sz = N_C * N_C * n3 * n2 * s3 * s3;
		dims[1] = N_C;
		dims[2] = n3;
		dims[3] = n2;
		dims[4] = s3;
		dims[5] = s3;
		n_dims = 6;
	}else if(deriv_ind == 4){ // FL deriv
		output_sz = N_C * n3 * max_output_sz3 * max_output_sz3;
		dims[1] = N_C;
		dims[2] = n3;
		dims[3] = max_output_sz3;
		dims[4] = max_output_sz3;
		n_dims = 5;
	}
	
	///////////////////////////////// allocate output mem
	sum_res_in = (PyArrayObject *) PyArray_FromDims(n_dims, dims, NPY_FLOAT);
	sum_res = (float *) sum_res_in -> data;
	
	/////////////////////////////////// cuda mem
	if(sum_res_c[gpu_ind][l][deriv_flag] == 0){
		printf("output buffer is empty. call einsum_deriv_gpu() before trying to get return values\n");
		return NULL;
	}

	cudaThreadSynchronize();
	

	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("CUDA error: %s, %s, %i\n", cudaGetErrorString(err),__FILE__,__LINE__);
		return NULL;
	}
	
	err = cudaMemcpy(sum_res, sum_res_c[gpu_ind][l][deriv_flag], output_sz * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK
	
	cudaFree(sum_res_c[gpu_ind][l][deriv_flag]);
	sum_res_c[gpu_ind][l][deriv_flag] = 0;
	
	return PyArray_Return(sum_res_in);
}
