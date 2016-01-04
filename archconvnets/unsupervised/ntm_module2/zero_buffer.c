static PyObject *zero_buffer(PyObject *self, PyObject *args){
    cudaError_t err;
	int gpu_ind, buffer_ind;
	
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
	
    cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(BUFFER_SZ != 0){
		err = cudaMemset(GPU_BUFFER, 0, BUFFER_SZ);  MALLOC_ERR_CHECK
	}
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
