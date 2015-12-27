__global__ void point_wise_add_scalar_kernel(float * out, float * a, float scalar1, float scalar2, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		out[ind_g] = scalar1 * a[ind_g] + scalar2;
	}
}

static PyObject * point_wise_add_scalar(PyObject *self, PyObject *args){
	cudaError_t err;
	float scalar1, scalar2;
	int a_ind, gpu_ind, out_buffer_ind;
	
	if (!PyArg_ParseTuple(args, "iffii", &a_ind, &scalar1, &scalar2, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(a_ind >= N_BUFFERS || a_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect.\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][a_ind] == 0){
		printf("buffer has zero size\n");
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
	point_wise_add_scalar_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][a_ind], scalar1, scalar2,
		buffer_sz[gpu_ind][a_ind]/(sizeof(DATA_TYPE)));
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
