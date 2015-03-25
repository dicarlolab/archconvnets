static PyObject *return_buffer(PyObject *self, PyObject *args){
    cudaError_t err;
	cudnnStatus_t status;
	PyArrayObject *data_in = NULL;
	float *data;
	int gpu_ind, buffer_ind, dims[5];
	
	if (!PyArg_ParseTuple(args, "ii", &buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(buffer_ind >= N_BUFFERS || buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(data_buffers[gpu_ind][buffer_ind] == NULL){
		printf("buffer not initialized on this gpu\n");
		return NULL;
	}
	
    cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	cudnnSetStream(handle, streams[gpu_ind]);
	
	dims[0] = data_dims[0][gpu_ind][buffer_ind];
	dims[1] = data_dims[1][gpu_ind][buffer_ind];
	dims[2] = data_dims[2][gpu_ind][buffer_ind];
	dims[3] = data_dims[3][gpu_ind][buffer_ind];
	
	data_in = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	data = (float *) data_in -> data;
	
	err = (cudaError_t)cudaMemcpy(data, data_buffers[gpu_ind][buffer_ind], dims[0]*dims[1]*dims[2]*dims[3] * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK
	
	cudnnSetStream(handle, NULL);
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	return PyArray_Return(data_in);
}
