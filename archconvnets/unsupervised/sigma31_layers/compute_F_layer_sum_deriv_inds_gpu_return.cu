static PyObject *compute_F_layer_sum_deriv_inds_gpu_return(PyObject *self, PyObject *args){
	cudaError_t err;
		
	int layer_ind, gpu_ind;
	PyArrayObject *F_sum_in;
	
	if (!PyArg_ParseTuple(args, "ii",  &layer_ind, &gpu_ind)) return NULL;

	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
		
	if(layer_ind < 0 || layer_ind > 4){
		printf("layer index (%i) not supported\n", layer_ind);
		return NULL;
	}
	
	if(F_sum_c[gpu_ind][layer_ind] == 0){
		printf("F_sum buffer not initialized, run compute_F_layer_sum_deriv_inds_gpu() before calling this function\n");
		return NULL;
	}
	
	F_sum_in = (PyArrayObject *) PyArray_FromDims(4, dims_F_sum[gpu_ind][layer_ind], NPY_FLOAT);
	float * F_sum = (float *) F_sum_in -> data;
	
	cudaThreadSynchronize();
	cudaMemcpy(F_sum, F_sum_c[gpu_ind][layer_ind], dims_F_sum[gpu_ind][layer_ind][0]*dims_F_sum[gpu_ind][layer_ind][1]*dims_F_sum[gpu_ind][layer_ind][2]*dims_F_sum[gpu_ind][layer_ind][3]*DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  CHECK_CUDA_ERR
	
	cudaFree(F_sum_c[gpu_ind][layer_ind]);
	cudaFree(F_partial_c[gpu_ind][layer_ind]);
	
	F_sum_c[gpu_ind][layer_ind] = 0;
	F_partial_c[gpu_ind][layer_ind] = 0;
	
	return PyArray_Return(F_sum_in);
}
