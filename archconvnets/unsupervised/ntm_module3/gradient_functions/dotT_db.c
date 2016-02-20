#define ADD_MEM_DADD_OUT_NUMEL (dim_above * n_imgs * C * mem_length)
#define ADD_MEM_DADD_OUT_SZ (ADD_MEM_DADD_OUT_NUMEL*sizeof(DATA_TYPE))

/*def add_mem_dadd_out(gw):
	temp = np.zeros((M, mem_length, C, mem_length),dtype='single')
	temp[:,range(mem_length),:,range(mem_length)] = gw.T
	return temp*/

// gw = (16, 6)  add_out = (16, 8)
// img, C, M    ....            img, C, mem_length

// deriv_above = (a, img, M, mem_length)
// deriv_above (a, img, M, mem_length) * gw (img, C, M) = [a, C, mem_length]

// batch a*img:
//		out[a,img] = gw[img] * deriv_above[a, img]


static PyObject *dotT_db(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, gw_ind, out_buffer_ind, deriv_above_ind;
	PyObject *gw_shape, *add_out_shape;
	
	if (!PyArg_ParseTuple(args, "iO!O!iii", &gw_ind, &PyTuple_Type, &gw_shape, &PyTuple_Type, &add_out_shape, 
		&deriv_above_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || gw_ind >= N_BUFFERS || gw_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(gw_shape, 0));
	long C = PyLong_AsLong(PyTuple_GetItem(gw_shape, 1));
	long M = PyLong_AsLong(PyTuple_GetItem(gw_shape, 2));
	
	long C2 = PyLong_AsLong(PyTuple_GetItem(add_out_shape, 1));
	long mem_length = PyLong_AsLong(PyTuple_GetItem(add_out_shape, 2));
	
	long dim_above = buffer_sz[gpu_ind][deriv_above_ind] / (n_imgs*M*mem_length*sizeof(DATA_TYPE));
	
	if(C != C2){
		printf("inner dot product dimensions do not match\n");
		return NULL;
	}
	
	if(n_imgs*C*M*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][gw_ind]){
		printf("specified input sizes do not equal to stored gpu buffer. %s\n", __FILE__);
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, ADD_MEM_DADD_OUT_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = ADD_MEM_DADD_OUT_SZ;
	}else if(ADD_MEM_DADD_OUT_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// deriv_above (a, img, M, mem_length) * gw (img, C, M) = [a, C, mem_length]

	// batch a*img:
	//		out[a,img] = gw[img] * deriv_above[a, img]
	
	cublasStatus_t err_blas;
	
	const float alpha = 1.0, beta = 0.0;
	
	/*for(int img = 0; img < n_imgs; img++){
		for(int a = 0; a < dim_above; a++){
			
			err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_N, mem_length, C, 
							M, &alpha,
                           gpu_buffers[gpu_ind][deriv_above_ind] + img*dim_above*M*mem_length + a*M*mem_length,
						   mem_length,
						   gpu_buffers[gpu_ind][gw_ind] + img*C*M, 
						   M, &beta, GPU_BUFFER_OUT + img*dim_above*C*mem_length + a*C*mem_length, mem_length);
			
			ERR_CHECK_BLAS
		}
	}*/
	
	int n_batches = n_imgs * dim_above;
	
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
	
	int batch = 0;
	for(int img = 0; img < n_imgs; img++){
		for(int a = 0; a < dim_above; a++){
			buffer1_pointers[batch] = gpu_buffers[gpu_ind][gw_ind] + img*C*M;
			buffer2_pointers[batch] = gpu_buffers[gpu_ind][deriv_above_ind] + img*dim_above*M*mem_length + a*M*mem_length;
			out_pointers[batch] = GPU_BUFFER_OUT + img*dim_above*C*mem_length + a*C*mem_length;
			
			batch ++;
		}
	}
	
	cudaMemcpy(buffer1_pointers_gpu, buffer1_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
	cudaMemcpy(buffer2_pointers_gpu, buffer2_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
	cudaMemcpy(out_pointers_gpu, out_pointers, n_batches * sizeof(float *), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
	
	err_blas = cublasSgemmBatched(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_N, mem_length, C, M, &alpha,
			 (const float**) buffer2_pointers_gpu, mem_length, (const float**) buffer1_pointers_gpu, M, &beta, out_pointers_gpu, mem_length, n_batches);
	
	cudaFree(buffer1_pointers_gpu);
	cudaFree(buffer2_pointers_gpu);
	cudaFree(out_pointers_gpu);
	
	free(buffer1_pointers);
	free(buffer2_pointers);
	free(out_pointers);
	
	ERR_CHECK_BLAS
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
