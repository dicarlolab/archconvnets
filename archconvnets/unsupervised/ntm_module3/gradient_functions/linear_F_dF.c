#define DLDF_NUMEL (deriv_above_dim0*x_dim0)
#define DLDF_SZ (DLDF_NUMEL*sizeof(DATA_TYPE))
#define X_SZ buffer_sz[gpu_ind][x_ind]

static PyObject * linear_F_dF(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *x_shape, *deriv_above_shape;
	int x_ind, deriv_above_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &x_ind, &PyTuple_Type, &x_shape, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(x_ind >= N_BUFFERS || x_ind < 0 || deriv_above_ind >=N_BUFFERS || deriv_above_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(X_SZ == 0 || buffer_sz[gpu_ind][deriv_above_ind] == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long deriv_above_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)deriv_above_shape,0));
	long deriv_above_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)deriv_above_shape,1));
	long x_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,0));
	long x_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,1));
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DLDF_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DLDF_SZ;
	}else if(DLDF_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	const float alpha = 1.0, beta = 0.0;
	
	// dot(deriv_above, x.T)
	
	cublasStatus_t err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_T, CUBLAS_OP_N, x_dim0, deriv_above_dim0, deriv_above_dim1, &alpha,
        gpu_buffers[gpu_ind][x_ind], x_dim1, gpu_buffers[gpu_ind][deriv_above_ind], deriv_above_dim1, &beta, GPU_BUFFER_OUT, x_dim0);
	
	ERR_CHECK_BLAS
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
