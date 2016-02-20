#define DLDF_SZ (keep_sum*deriv_above_dim0*x_dim0*sizeof(DATA_TYPE))

__global__ void pairwise_product_kernel_dF(float * deriv_above, float * X, float * out_buffer, 
		int dim_above, int dim2, int dim0, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = data_out_numel / THREAD_CAPACITY;
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, r, img, a, d2, d0;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		// we are computing the output data_out[img, a, dim2, dim0]
		img = ind_g / (dim_above*dim2*dim0);
		r = ind_g % (dim_above*dim2*dim0);
		
		a = r / (dim2*dim0);
		r = r % (dim2*dim0);
		
		d2 = r / dim0;
		d0 = r % dim0;
		
		// out[img, a, d2, d0] = deriv_above[img, a, d2] * X[img, d0]
		out_buffer[ind_g] = deriv_above[img*dim_above*dim2 + a*dim2 + d2] * X[img*dim0 + d0];
	}
}

static PyObject * linear_F_dF(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *x_shape, *deriv_above_shape;
	int x_ind, deriv_above_ind, out_buffer_ind, gpu_ind, keep_sum;
	
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &x_ind, &PyTuple_Type, &x_shape, &deriv_above_ind, &PyTuple_Type, 
			&deriv_above_shape, &out_buffer_ind, &gpu_ind)) 
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
	
	//[img,out1,out2]
	
	// get sizes
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape, 1));
	long deriv_above_dim0 = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape, 2));
	long deriv_above_dim1 = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape, 3));
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(x_shape, 0));
	long x_dim0 = PyLong_AsLong(PyTuple_GetItem(x_shape, 1));
	long x_dim1 = PyLong_AsLong(PyTuple_GetItem(x_shape, 2));
	
	// sum all images together or not
	if(dim_above == 1)
		keep_sum = 1;
	else
		keep_sum = n_imgs*dim_above;
	

	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DLDF_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DLDF_SZ;
	}else if(DLDF_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	const float alpha = 1;
	cublasStatus_t err_blas;
	
	// dot(deriv_above, x.T)
	
	////////////// case where cublas can perform as a matrix multiplication
	if(x_dim1 != 1){ // x_dim != 1
		
		//sum all images together.... a custom kernel might be helpful here...cannot simply use cublasSgemmBatched because this is an atomic op.
		if(dim_above == 1){ 
			const float beta = 1; // incremental adds..
			err = cudaMemset(GPU_BUFFER_OUT, 0, DLDF_SZ);  MALLOC_ERR_CHECK
		
			for(int img = 0; img < n_imgs; img++){
				for(int batch = 0; batch < dim_above; batch++){
					err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_T, CUBLAS_OP_N, x_dim0, deriv_above_dim0, deriv_above_dim1, &alpha,
						gpu_buffers[gpu_ind][x_ind] + img*x_dim0*x_dim1, x_dim1, 
						gpu_buffers[gpu_ind][deriv_above_ind] + img*dim_above*deriv_above_dim0*deriv_above_dim1 + batch*deriv_above_dim0*deriv_above_dim1, 
						deriv_above_dim1, &beta, GPU_BUFFER_OUT, x_dim0);
				}
			}
		}else{
			////////////// do not sum images/dim_above together
			const float beta = 0; // ignore/overwrite whatever is in out_buffer
			int n_batches = n_imgs*dim_above;
			
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
			
			int gpu_batch = 0;
			for(int img = 0; img < n_imgs; img++){
				for(int batch = 0; batch < dim_above; batch++){
					buffer1_pointers[gpu_batch] = gpu_buffers[gpu_ind][deriv_above_ind] + img*dim_above*deriv_above_dim0*deriv_above_dim1 + batch*deriv_above_dim0*deriv_above_dim1;
					buffer2_pointers[gpu_batch] = gpu_buffers[gpu_ind][x_ind] + img*x_dim0*x_dim1;
					out_pointers[gpu_batch] = GPU_BUFFER_OUT + img*dim_above*deriv_above_dim0*x_dim0 + batch*deriv_above_dim0*x_dim0;
					
					gpu_batch ++;
				}
			}
			
			cudaMemcpy(buffer1_pointers_gpu, buffer1_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
			cudaMemcpy(buffer2_pointers_gpu, buffer2_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
			cudaMemcpy(out_pointers_gpu, out_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
			
			err_blas = cublasSgemmBatched(handle_blas[gpu_ind], CUBLAS_OP_T, CUBLAS_OP_N, x_dim0, deriv_above_dim0, deriv_above_dim1, &alpha,
					(const float**) buffer2_pointers_gpu, x_dim1, (const float**) buffer1_pointers_gpu, 
					deriv_above_dim1, &beta, out_pointers_gpu, x_dim0, n_batches);
			
			///////////////////////////////////////////// possible race condition if sync isn't present
			cudaFree(buffer1_pointers_gpu);
			cudaFree(buffer2_pointers_gpu);
			cudaFree(out_pointers_gpu);
			
			free(buffer1_pointers);
			free(buffer2_pointers);
			free(out_pointers);
			
		}
		
		ERR_CHECK_BLAS
		
	}else if(dim_above == 1){ // x_dim1 = 1, dim_above=1
		const float beta = 0;
		
		////////////////////// sum across images and dim_above: reduces the problem down to a single matrix multiply:
		
		// X: [img,dim0]; deriv_above: [img,dim2].... compute dot(deriv_above.T, X)
		
		err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_T, x_dim0, deriv_above_dim0, n_imgs, &alpha,
			 gpu_buffers[gpu_ind][x_ind], x_dim0, gpu_buffers[gpu_ind][deriv_above_ind], deriv_above_dim0, &beta, 
			 GPU_BUFFER_OUT, x_dim0);
			 
		ERR_CHECK_BLAS
		
	}else{ // x_dim = 1, dim_above != 1.... do *not* sum across imgs/dim_above
	
		///////// case where cublas can't handle [dim1,1] * [1,dim2]
		// compute: out[img,a,dim2,dim0] = dim_above[img,a,dim2] * X[img,dim0]
		
		// determine number of blocks
		int n_blocks = (int)ceil((double)DLDF_SZ/(sizeof(DATA_TYPE)*MAX_THREADS_PER_BLOCK));
		if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
		
		pairwise_product_kernel_dF <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][deriv_above_ind], 
			gpu_buffers[gpu_ind][x_ind], GPU_BUFFER_OUT, 
			dim_above, deriv_above_dim0, x_dim0, DLDF_SZ/sizeof(DATA_TYPE));
		
		CHECK_CUDA_ERR
		
	}
	
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
