// out_buffer[img] = a[img] + b

static PyObject * add_points_batch(PyObject *self, PyObject *args){
	cudaError_t err;
	int a_ind, b_ind, gpu_ind, out_buffer_ind;
	
	if (!PyArg_ParseTuple(args, "iiii", &a_ind, &b_ind, &out_buffer_ind, &gpu_ind)) 
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
	
	int n_imgs = buffer_sz[gpu_ind][a_ind] / buffer_sz[gpu_ind][b_ind];
	if(buffer_sz[gpu_ind][a_ind] % buffer_sz[gpu_ind][b_ind] != 0){
		printf("b must be multiple of a, %s\n", __FILE__);
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][a_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][a_ind];
	}else if(buffer_sz[gpu_ind][a_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaMemcpy(gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][a_ind], OUT_BUFFER_SZ, cudaMemcpyDeviceToDevice); CHECK_CUDA_ERR
	
	const float scalar = 1;
	cublasStatus_t err_blas;
	
	int b_sz = buffer_sz[gpu_ind][b_ind] / sizeof(DATA_TYPE);
	
	// perform add: [better way to batch?]
	for(int img = 0; img < n_imgs; img++){
		err_blas = cublasSaxpy(handle_blas[gpu_ind], b_sz, &scalar, gpu_buffers[gpu_ind][b_ind], 1, 
				gpu_buffers[gpu_ind][out_buffer_ind] + img*b_sz, 1); ERR_CHECK_BLAS
	}
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
