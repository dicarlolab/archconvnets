__global__ void linear_F_dF_kernel(float * x, float * dldf, int F_dim0, int x_dim0, int x_dim1){ 
	int j = threadIdx.x / x_dim0;
	int k = threadIdx.x % x_dim0;

	for(int i = 0; i < F_dim0; i++){
		for(int i_local = 0; i_local < F_dim0; i_local++){
			if(i_local == i)
				DLDF(i,j,i,k) = X(k,j);
			else
				DLDF(i,j,i_local,k) = 0;
		}
	}

	return;
}

static PyObject * mult_partials_collapse(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *da_dc_shape;
	int a_ndim, da_dc_ind, out_buffer_ind, gpu_ind;
	
	// mult_partials_collapse(out_buffer_ind, a_ndim, da_dc_shape, out_buffer_ind2, gpu_ind)
	
	if (!PyArg_ParseTuple(args, "iiO!ii", &da_dc_ind, &a_ndim, &PyTuple_Type, &da_dc_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(da_dc_ind >= N_BUFFERS || da_dc_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
		printf("buffer index incorrect.\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][x_ind] == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long F_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)F_shape,0));
	long F_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)F_shape,1));
	long x_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,0));
	long x_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,1));
	
	if(x_dim0*x_dim1*sizeof(DATA_TYPE) != X_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DLDF_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DLDF_SZ;
	}else if(DLDF_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	linear_F_dF_kernel <<< 1, x_dim1 * x_dim0 >>> (gpu_buffers[gpu_ind][x_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], F_dim0, x_dim0, x_dim1);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
