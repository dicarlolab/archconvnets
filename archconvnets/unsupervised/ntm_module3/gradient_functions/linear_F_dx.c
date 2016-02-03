#define DLDX_SZ (deriv_above_dim0*x_dim0*x_dim1*sizeof(DATA_TYPE))

static PyObject * linear_F_dx(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *deriv_above_shape, *F_shape,*x_shape;
	int F_ind, deriv_above_ind, out_buffer_ind, gpu_ind;
	const float alpha = 1.0, beta = 0.0;
	
	if (!PyArg_ParseTuple(args, "iO!O!iO!ii", &F_ind, &PyTuple_Type, &F_shape, &PyTuple_Type, &x_shape, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape,
			&out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(F_ind >= N_BUFFERS || F_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			deriv_above_ind >= N_BUFFERS || deriv_above_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long x_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,0));
	long x_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,1));
	long F_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)F_shape,0));
	long F_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)F_shape,1));
	long deriv_above_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)deriv_above_shape,0));
	long deriv_above_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)deriv_above_shape,1));
	long deriv_above_dim2 = PyLong_AsLong(PyTuple_GetItem((PyObject *)deriv_above_shape,2));
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DLDX_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DLDX_SZ;
	}else if(DLDX_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	///////////////////////
	// non batched version
	/*cublasStatus_t err_blas;
	for(int batch = 0; batch < deriv_above_dim0; batch++){
		err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_T, deriv_above_dim2, F_dim1, F_dim0, &alpha,
                                 gpu_buffers[gpu_ind][deriv_above_ind]+ batch*deriv_above_dim1*deriv_above_dim2, deriv_above_dim2, gpu_buffers[gpu_ind][F_ind], F_dim1, &beta, gpu_buffers[gpu_ind][out_buffer_ind] + batch*x_dim0*x_dim1, deriv_above_dim2);
		ERR_CHECK_BLAS
	}*/
	
	////////////////////////////////////////////////////////
	// setup batch pointers
	float ** F_pointers = (float **) malloc(deriv_above_dim0 * sizeof(float *));
	float ** deriv_above_pointers = (float **) malloc(deriv_above_dim0 * sizeof(float *));
	float ** out_pointers = (float **) malloc(deriv_above_dim0 * sizeof(float *));	
	
	if(F_pointers == NULL || deriv_above_pointers == NULL || out_pointers == NULL){
		printf("malloc err line: %i\n",__LINE__);
		return NULL;
	}
	
	float ** F_pointers_gpu, ** deriv_above_pointers_gpu, **out_pointers_gpu;
	
	err = cudaMalloc((void**) &F_pointers_gpu, deriv_above_dim0*sizeof(float*)); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &deriv_above_pointers_gpu, deriv_above_dim0*sizeof(float*)); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &out_pointers_gpu, deriv_above_dim0*sizeof(float*)); MALLOC_ERR_CHECK
	
	for(int batch = 0; batch < deriv_above_dim0; batch++){
		F_pointers[batch] = gpu_buffers[gpu_ind][F_ind];
		deriv_above_pointers[batch] = gpu_buffers[gpu_ind][deriv_above_ind] + batch*deriv_above_dim1*deriv_above_dim2;
		out_pointers[batch] = gpu_buffers[gpu_ind][out_buffer_ind] + batch*x_dim0*x_dim1;
	}
	
	cudaMemcpy(F_pointers_gpu, F_pointers, deriv_above_dim0 * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
	cudaMemcpy(deriv_above_pointers_gpu, deriv_above_pointers, deriv_above_dim0 * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
	cudaMemcpy(out_pointers_gpu, out_pointers, deriv_above_dim0 * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
	
	cublasStatus_t err_blas = cublasSgemmBatched(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_T, deriv_above_dim2, F_dim1, F_dim0, &alpha,
                                 (const float**) deriv_above_pointers_gpu, deriv_above_dim2,(const float**) F_pointers_gpu, F_dim1, &beta, out_pointers_gpu, deriv_above_dim2, deriv_above_dim0);
	

	ERR_CHECK_BLAS
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	///////////////////////////////////////////// possible race condition if sync isn't present
	cudaFree(F_pointers_gpu);
	cudaFree(deriv_above_pointers_gpu);
	cudaFree(out_pointers_gpu);
	
	free(F_pointers);
	free(deriv_above_pointers);
	free(out_pointers);
	
	Py_INCREF(Py_None);
	return Py_None;
}
