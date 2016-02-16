static PyObject * copy_buffer(PyObject *self, PyObject *args){
	cudaError_t err;
	int b_ind, gpu_ind, out_buffer_ind;
	
	if (!PyArg_ParseTuple(args, "iii", &b_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || b_ind >= N_BUFFERS || b_ind < 0){ 
		printf("buffer index incorrect.\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][b_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][b_ind];
		
	}else if(buffer_sz[gpu_ind][b_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size %li %li %s\n", buffer_sz[gpu_ind][b_ind], OUT_BUFFER_SZ, __FILE__);
		return NULL;
	}
	
	cudaMemcpy(gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][b_ind], OUT_BUFFER_SZ, cudaMemcpyDeviceToDevice); CHECK_CUDA_ERR
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
