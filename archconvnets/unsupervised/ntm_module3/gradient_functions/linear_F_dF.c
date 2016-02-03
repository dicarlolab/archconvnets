#define DLDF_SZ (n_batches*deriv_above_dim0*x_dim0*sizeof(DATA_TYPE)/n_imgs)

static PyObject * linear_F_dF(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *x_shape, *deriv_above_shape;
	int x_ind, deriv_above_ind, out_buffer_ind, gpu_ind, n_batches;
	
	if (!PyArg_ParseTuple(args, "iO!iO!iii", &x_ind, &PyTuple_Type, &x_shape, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape, 
			&out_buffer_ind, &n_batches, &gpu_ind)) 
		return NULL;
    
	if(x_ind >= N_BUFFERS || x_ind < 0 || deriv_above_ind >= N_BUFFERS || deriv_above_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][x_ind] == 0 || buffer_sz[gpu_ind][deriv_above_ind] == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// deriv_above: (2, 3, 5, 10, 4); F: (10,3)
	// x_reshaped (5, 3, 4) deriv_above_reshaped (30, 10, 4), n_batches=30 ==== n_dims_not_sum_prod/n_imgs
	// compute: (2*3, 10,3)
	// (sum over all imgs, [dim_size: 5])
	
	// get sizes
	int dim_offset = 0, n_imgs = 1;
	if(n_batches > 1){ 
		dim_offset = 1; // then first dimension is deriv_above not summed dims and imgs
		n_imgs = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,0));
		
		if(n_batches % n_imgs != 0){
			printf("deriv above or n_imgs incorrect\n");
			return NULL;
		}
	}
	
	long deriv_above_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)deriv_above_shape,dim_offset));
	long deriv_above_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)deriv_above_shape,dim_offset+1));
	long x_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,dim_offset));
	long x_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,dim_offset+1));
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DLDF_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DLDF_SZ;
	}else if(DLDF_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	const float alpha = 1.0;
	
	cublasStatus_t err_blas;
	
	// dot(deriv_above, x.T)
	if(n_batches == 1){
		const float beta = 0.0;
		err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_T, CUBLAS_OP_N, x_dim0, deriv_above_dim0, deriv_above_dim1, &alpha,
			gpu_buffers[gpu_ind][x_ind], x_dim1, gpu_buffers[gpu_ind][deriv_above_ind], deriv_above_dim1, &beta, GPU_BUFFER_OUT, x_dim0);
	}else{
		int img;
		const float beta = 1; // sum values
		
		err = cudaMemset(GPU_BUFFER_OUT, 0, DLDF_SZ);  MALLOC_ERR_CHECK
		
		/////////////////////////////////////////////////
		// non-batched version
		for(int batch = 0; batch < n_batches; batch++){
			img = batch % n_imgs; // batch = d_above_ind*n_imgs + img_ind;
			err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_T, CUBLAS_OP_N, x_dim0, deriv_above_dim0, deriv_above_dim1, &alpha,
				gpu_buffers[gpu_ind][x_ind] + img*x_dim0*x_dim1, x_dim1, gpu_buffers[gpu_ind][deriv_above_ind] + batch*deriv_above_dim0*deriv_above_dim1, 
					deriv_above_dim1, &beta, GPU_BUFFER_OUT, x_dim0);
		}
		
		// the following doesn't work, likely because I'm trying to write all batches to the same buffer, which seemingly isn't
		// done in an atomic way
		/*////////////////////////////////////////////////////////
		// setup batch pointers
		float ** x_pointers = (float **) malloc(n_batches * sizeof(float *));
		float ** deriv_above_pointers = (float **) malloc(n_batches * sizeof(float *));
		float ** out_pointers = (float **) malloc(n_batches * sizeof(float *));	
		
		if(x_pointers == NULL || deriv_above_pointers == NULL || out_pointers == NULL){
			printf("malloc err line: %i\n",__LINE__);
			return NULL;
		}
		
		float ** x_pointers_gpu, ** deriv_above_pointers_gpu, **out_pointers_gpu;
		
		err = cudaMalloc((void**) &x_pointers_gpu, n_batches*sizeof(float*)); MALLOC_ERR_CHECK
		err = cudaMalloc((void**) &deriv_above_pointers_gpu, n_batches*sizeof(float*)); MALLOC_ERR_CHECK
		err = cudaMalloc((void**) &out_pointers_gpu, n_batches*sizeof(float*)); MALLOC_ERR_CHECK
		
		for(int batch = 0; batch < n_batches; batch++){
			img = batch % n_imgs; // batch = d_above_ind*n_imgs + img_ind;
			x_pointers[batch] = gpu_buffers[gpu_ind][x_ind] + img*x_dim0*x_dim1;
			deriv_above_pointers[batch] = gpu_buffers[gpu_ind][deriv_above_ind] + batch*deriv_above_dim0*deriv_above_dim1;
			out_pointers[batch] = GPU_BUFFER_OUT;
		}
		
		cudaMemcpy(x_pointers_gpu, x_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
		cudaMemcpy(deriv_above_pointers_gpu, deriv_above_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
		cudaMemcpy(out_pointers_gpu, out_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
		
		err_blas = cublasSgemmBatched(handle_blas[gpu_ind], CUBLAS_OP_T, CUBLAS_OP_N, x_dim0, deriv_above_dim0, deriv_above_dim1, &alpha,
						(const float**) x_pointers_gpu, x_dim1, (const float**) deriv_above_pointers_gpu, deriv_above_dim1, &beta, out_pointers_gpu, x_dim0, n_batches);
		
		///////////////////////////////////////////// possible race condition if sync isn't present
		cudaFree(x_pointers_gpu);
		cudaFree(deriv_above_pointers_gpu);
		cudaFree(out_pointers_gpu);
		
		free(x_pointers);
		free(deriv_above_pointers);
		free(out_pointers);*/
		
	}
	
	ERR_CHECK_BLAS
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
