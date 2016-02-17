#define GPU_BUFFER1 gpu_buffers[gpu_ind][buffer_ind1]
#define GPU_BUFFER2 gpu_buffers[gpu_ind][buffer_ind2]
#define BUFFER_SZ1 buffer_sz[gpu_ind][buffer_ind1]
#define BUFFER_SZ2 buffer_sz[gpu_ind][buffer_ind2]

#define DATA_OUT_SZ (n_batches*buffer1_dim1*buffer2_dim2*sizeof(DATA_TYPE))

static PyObject *dot_batched(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, buffer_ind1, buffer_ind2, out_buffer_ind;
	PyObject *buffer_shape1, *buffer_shape2;
	
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &buffer_ind1, &PyTuple_Type, &buffer_shape1, &buffer_ind2, 
			&PyTuple_Type, &buffer_shape2, &out_buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(buffer_ind1 >= N_BUFFERS || buffer_ind1 < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || 
			buffer_ind2 >= N_BUFFERS || buffer_ind2 < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long n_batches = PyLong_AsLong(PyTuple_GetItem(buffer_shape1,0));
	long buffer1_dim1 = PyLong_AsLong(PyTuple_GetItem(buffer_shape1,1));
	long buffer1_dim2 = PyLong_AsLong(PyTuple_GetItem(buffer_shape1,2));
	
	long buffer2_dim1 = PyLong_AsLong(PyTuple_GetItem(buffer_shape2,1));
	long buffer2_dim2 = PyLong_AsLong(PyTuple_GetItem(buffer_shape2,2));
	
	if(buffer1_dim2 != buffer2_dim1){
		printf("inner dot product dimensions do not match, (%li, %li), (%li, %li)\n", buffer1_dim1, buffer1_dim2, buffer2_dim1, buffer2_dim2);
		return NULL;
	}
	
	if(n_batches*buffer1_dim1*buffer1_dim2*sizeof(DATA_TYPE) != BUFFER_SZ1 || n_batches*buffer2_dim1*buffer2_dim2*sizeof(DATA_TYPE) != BUFFER_SZ2){
		printf("specified input sizes do not equal to stored gpu buffer. %s\n", __FILE__);
		printf("%li %li %li %li", n_batches*buffer1_dim1*buffer1_dim2*sizeof(DATA_TYPE), BUFFER_SZ1, n_batches*buffer2_dim1*buffer2_dim2*sizeof(DATA_TYPE), BUFFER_SZ2);
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DATA_OUT_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DATA_OUT_SZ;
	}else if(DATA_OUT_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	const float alpha = 1.0, beta = 0.0;
	
	cublasStatus_t err_blas;
	
	if(n_batches == 1){
		err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_N, buffer2_dim2, buffer1_dim1, buffer1_dim2, &alpha,
                           GPU_BUFFER2, buffer2_dim2, GPU_BUFFER1, buffer1_dim2, &beta, GPU_BUFFER_OUT, buffer2_dim2);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}else{
		
		/// non-batched version:
		/*for(int batch = 0; batch < n_batches; batch++){
			err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_N, buffer2_dim2, buffer1_dim1, buffer1_dim2, &alpha,
									 GPU_BUFFER2 + batch*buffer2_dim1*buffer2_dim2, buffer2_dim2, GPU_BUFFER1, buffer1_dim2, &beta, GPU_BUFFER_OUT + batch*buffer1_dim1*buffer2_dim2, buffer2_dim2);
			ERR_CHECK_BLAS
		}*/
		
		
		////////////////////////////////////////////////////////
		// setup batch pointers on CPU, then copy to GPU
		float ** buffer1_pointers = (float **) malloc(n_batches * sizeof(float *));
		float ** buffer2_pointers = (float **) malloc(n_batches * sizeof(float *));
		float ** out_pointers = (float **) malloc(n_batches * sizeof(float *));	
		
		if(buffer1_pointers == NULL || buffer2_pointers == NULL || out_pointers == NULL){
			printf("malloc err line: %i, %s\n",__LINE__,__FILE__);
			return NULL;
		}
		
		float ** buffer1_pointers_gpu, ** buffer2_pointers_gpu, **out_pointers_gpu;
		
		err = cudaMalloc((void**) &buffer1_pointers_gpu, n_batches*sizeof(float*)); MALLOC_ERR_CHECK
		err = cudaMalloc((void**) &buffer2_pointers_gpu, n_batches*sizeof(float*)); MALLOC_ERR_CHECK
		err = cudaMalloc((void**) &out_pointers_gpu, n_batches*sizeof(float*)); MALLOC_ERR_CHECK
		
		for(int batch = 0; batch < n_batches; batch++){
			buffer1_pointers[batch] = GPU_BUFFER1 + batch*buffer1_dim1*buffer2_dim2;
			buffer2_pointers[batch] = GPU_BUFFER2 + batch*buffer2_dim1*buffer2_dim2;
			out_pointers[batch] = GPU_BUFFER_OUT + batch*buffer1_dim1*buffer2_dim2;
		}
		
		cudaMemcpy(buffer1_pointers_gpu, buffer1_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
		cudaMemcpy(buffer2_pointers_gpu, buffer2_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
		cudaMemcpy(out_pointers_gpu, out_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
		
		err_blas = cublasSgemmBatched(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_N, buffer2_dim2, buffer1_dim1, buffer1_dim2, &alpha,
									 (const float**) buffer2_pointers_gpu, buffer2_dim2, (const float**) buffer1_pointers_gpu, buffer1_dim2, &beta, out_pointers_gpu, buffer2_dim2, n_batches);
		
		///////////////////////////////////////////// possible race condition if sync isn't present
		cudaFree(buffer1_pointers_gpu);
		cudaFree(buffer2_pointers_gpu);
		cudaFree(out_pointers_gpu);
		
		free(buffer1_pointers);
		free(buffer2_pointers);
		free(out_pointers);
	}
	
	ERR_CHECK_BLAS
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
