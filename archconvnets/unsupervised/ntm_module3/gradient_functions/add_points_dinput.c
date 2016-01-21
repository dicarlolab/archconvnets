static PyObject * add_points_dinput(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, out_buffer_ind, deriv_above_ind;
	float scalar;
	PyObject *a_shape;
	
	if (!PyArg_ParseTuple(args, "O!iifi", &PyTuple_Type, &a_shape, &out_buffer_ind, &deriv_above_ind, &scalar, &gpu_ind)) 
		return NULL;
    
	if(out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || deriv_above_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect.\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long a_dim0 = PyLong_AsLong(PyTuple_GetItem(a_shape,0));
	long a_dim1 = PyLong_AsLong(PyTuple_GetItem(a_shape,1));
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][deriv_above_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][deriv_above_ind];
	}else if(OUT_BUFFER_SZ != buffer_sz[gpu_ind][deriv_above_ind]){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaMemcpy(gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][deriv_above_ind], buffer_sz[gpu_ind][out_buffer_ind], cudaMemcpyDeviceToDevice); CHECK_CUDA_ERR
	
	if(scalar != 1){
		cublasStatus_t err_blas = cublasSscal(handle_blas[gpu_ind], buffer_sz[gpu_ind][out_buffer_ind]/sizeof(DATA_TYPE), &scalar, gpu_buffers[gpu_ind][out_buffer_ind], 1);
		ERR_CHECK_BLAS
	}
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
