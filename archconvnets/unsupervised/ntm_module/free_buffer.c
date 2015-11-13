static PyObject *free_buffer(PyObject *self, PyObject *args){
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
	
	if(BUFFER_SZ != 0){
		cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
		err = cudaFree((void**) GPU_BUFFER); CHECK_CUDA_ERR
		BUFFER_SZ = 0;
		cudaSetDevice(0); CHECK_CUDA_ERR
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}