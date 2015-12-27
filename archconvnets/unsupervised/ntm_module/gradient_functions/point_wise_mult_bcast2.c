__global__ void point_wise_mult_bcast2_kernel(float * out, float * a, float * b, float scalar, int dim2_dim3_dim4, int dim3_dim4, int dim2, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, i,j, remainder;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		i = ind_g / dim2_dim3_dim4;
		remainder = ind_g % dim2_dim3_dim4;
		
		j = remainder / dim3_dim4;
		
		out[ind_g] = a[ind_g] * b[i*dim2 + j] * scalar;
	}
}

static PyObject * point_wise_mult_bcast2(PyObject *self, PyObject *args){
	cudaError_t err;
	float scalar;
	int a_ind, b_ind, gpu_ind, out_buffer_ind;
	PyObject *a_shape;
	
	if (!PyArg_ParseTuple(args, "iO!ifii", &a_ind, &PyTuple_Type, &a_shape, &b_ind, &scalar, &out_buffer_ind, &gpu_ind)) 
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
	
	// get size
	long dim1 = PyLong_AsLong(PyTuple_GetItem(a_shape,0));
	long dim2 = PyLong_AsLong(PyTuple_GetItem(a_shape,1));
	long dim3 = PyLong_AsLong(PyTuple_GetItem(a_shape,2));
	long dim4 = PyLong_AsLong(PyTuple_GetItem(a_shape,3));
	
	if(buffer_sz[gpu_ind][a_ind] != dim1*dim2*dim3*dim4*sizeof(DATA_TYPE) || buffer_sz[gpu_ind][b_ind] != dim1*dim2*sizeof(DATA_TYPE)){
		printf("buffer size does not match given size\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][a_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][a_ind];
	}else if(buffer_sz[gpu_ind][a_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)buffer_sz[gpu_ind][a_ind]/(sizeof(DATA_TYPE)*MAX_THREADS_PER_BLOCK));
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	point_wise_mult_bcast2_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][a_ind], 
		gpu_buffers[gpu_ind][b_ind], scalar, dim2*dim3*dim4, dim3*dim4, dim2, buffer_sz[gpu_ind][a_ind]/(sizeof(DATA_TYPE)));
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
