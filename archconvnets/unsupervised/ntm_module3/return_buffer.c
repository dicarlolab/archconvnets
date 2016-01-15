static PyObject *return_buffer(PyObject *self, PyObject *args){
    cudaError_t err;
	PyArrayObject *numpy_buffer_temp = NULL;
	float *data;
	int gpu_ind, buffer_ind, warn;
	
	if (!PyArg_ParseTuple(args, "iii", &buffer_ind, &warn, &gpu_ind)) 
		return NULL;
        
	if(buffer_ind >= N_BUFFERS || buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
    cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(BUFFER_SZ == 0){
		if(warn == 1)
			printf("buffer not initialized. return_buffer()\n");
		return NULL;
	}
	
	int dims[1];
	dims[0] = BUFFER_SZ/sizeof(DATA_TYPE);
	numpy_buffer_temp = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	if(numpy_buffer_temp == NULL){
		printf("couldnt create output numpy array\n");
		return NULL;
	}
	data = (float *) PyArray_DATA(numpy_buffer_temp);
	
	err = cudaMemcpy(data, GPU_BUFFER, BUFFER_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	return PyArray_Return(numpy_buffer_temp);
}
