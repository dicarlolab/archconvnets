/* out_buffer[img] = a[img] + b

 deriv_above: (2,3, 10, 4,5),   a: (10, 4,5)
 b: (4,5) ===> return (2,3, 4,5) (sum deriv_above across images):
	deriv_above(2*3, 10, 4*5) batch first dimension -> out[batch] = deriv_above[batch].T * (10,1)

 NOTE: add_points_batch_dinputA = add_points_dinput [with additional_args=[1]] */

 static PyObject * add_points_batch_dinputB(PyObject *self, PyObject *args){
	cudaError_t err;
	int a_ind, b_ind, gpu_ind, out_buffer_ind, deriv_above_ind;
	
	if (!PyArg_ParseTuple(args, "iiiii", &a_ind, &b_ind, &deriv_above_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(a_ind >= N_BUFFERS || a_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			deriv_above_ind >= N_BUFFERS || deriv_above_ind < 0 || b_ind >= N_BUFFERS || b_ind < 0){ 
		printf("buffer index incorrect.\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	int a_sz = buffer_sz[gpu_ind][a_ind] / sizeof(DATA_TYPE);
	int b_sz = buffer_sz[gpu_ind][b_ind] / sizeof(DATA_TYPE);
	
	int n_imgs = buffer_sz[gpu_ind][a_ind] / buffer_sz[gpu_ind][b_ind];
	if(buffer_sz[gpu_ind][a_ind] % buffer_sz[gpu_ind][b_ind] != 0){
		printf("b must be multiple of a, %s\n", __FILE__);
		return NULL;
	}
	
	int n_batches = buffer_sz[gpu_ind][deriv_above_ind] / buffer_sz[gpu_ind][a_ind];
	if(buffer_sz[gpu_ind][deriv_above_ind] % buffer_sz[gpu_ind][a_ind] != 0){
		printf("deriv_above must be multiple of a, %s\n", __FILE__);
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	unsigned intended_sz = n_batches * buffer_sz[gpu_ind][b_ind];
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, intended_sz); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = intended_sz;
	}else if(intended_sz != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cublasStatus_t err_blas;
	const float alpha = 1, beta = 0;
	float * ones_gpu, * ones;
	
	ones = (float *) malloc(n_imgs * sizeof(float));	
	for(int i = 0; i < n_imgs; i++){
		ones[i] = 1;
	}
	
	err = cudaMalloc((void**) &ones_gpu, n_imgs*sizeof(DATA_TYPE)); MALLOC_ERR_CHECK
	cudaMemcpy(ones_gpu, ones, n_imgs * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
	
	///////////// could be parallelized...
	for(int batch = 0; batch < n_batches; batch++){
		err_blas = cublasSgemv(handle_blas[gpu_ind], CUBLAS_OP_N,
                           b_sz, n_imgs, &alpha, gpu_buffers[gpu_ind][deriv_above_ind] + batch*a_sz, b_sz,
                           ones_gpu, 1, &beta, GPU_BUFFER_OUT + batch*b_sz, 1);
		ERR_CHECK_BLAS
	}
	
	cudaFree(ones_gpu);
	free(ones);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
