#define ADD_BLAS_SZ (buffer_sz[gpu_ind][a_ind]/(sizeof(DATA_TYPE)))

// out_buffer = a * scalar0 + b * scalar

static PyObject * point_wise_add(PyObject *self, PyObject *args){
	cudaError_t err;
	float scalar, scalar0;
	int a_ind, b_ind, gpu_ind, out_buffer_ind;
	char buffer_prev_init = 1;
	
	if (!PyArg_ParseTuple(args, "iiffii", &a_ind, &b_ind, &scalar, &scalar0, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(a_ind >= N_BUFFERS || a_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0
			|| b_ind >= N_BUFFERS || b_ind < 0){ 
		printf("buffer index incorrect.\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][b_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][b_ind];
		buffer_prev_init = 0;
	}else if(buffer_sz[gpu_ind][b_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][a_ind] != buffer_sz[gpu_ind][b_ind]){
		printf("buffer sizes are not equal %li, %li\n", buffer_sz[gpu_ind][a_ind], buffer_sz[gpu_ind][b_ind]);
		return NULL;
	}
	
	// if A has not been initialized, simply copy B to out buffer
	if(a_ind == out_buffer_ind && buffer_prev_init == 0){
		cudaMemcpy(gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][b_ind], OUT_BUFFER_SZ, cudaMemcpyDeviceToDevice);
	
	// perform add:
	}else{
		if(out_buffer_ind != a_ind)
			cudaMemcpy(gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][a_ind], buffer_sz[gpu_ind][a_ind], cudaMemcpyDeviceToDevice);
	
	
		if(scalar0 != 1)
			cublasSscal(handle_blas[gpu_ind], ADD_BLAS_SZ, &scalar0, gpu_buffers[gpu_ind][out_buffer_ind], 1);
		
		cublasSaxpy(handle_blas[gpu_ind], ADD_BLAS_SZ, &scalar, gpu_buffers[gpu_ind][b_ind], 1, gpu_buffers[gpu_ind][out_buffer_ind], 1);
	}
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
