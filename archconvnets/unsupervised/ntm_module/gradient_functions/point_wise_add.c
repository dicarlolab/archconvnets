__global__ void point_wise_add_kernel(float * a, float * b, float scalar, int data_out_numel){
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
		
		a[ind_g] += b[ind_g] * scalar;
	}
}

static PyObject * point_wise_add(PyObject *self, PyObject *args){
	cudaError_t err;
	float scalar;
	int a_ind, b_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iifi", &a_ind, &b_ind, &scalar, &gpu_ind)) 
		return NULL;
    
	if(a_ind >= N_BUFFERS || a_ind < 0 || b_ind >= N_BUFFERS || b_ind < 0){ 
		printf("buffer index incorrect.\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][a_ind] == 0){
		printf("buffer not initialized\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][a_ind] != buffer_sz[gpu_ind][b_ind]){
		printf("buffer sizes are not equal\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)buffer_sz[gpu_ind][a_ind]/(sizeof(DATA_TYPE)*MAX_THREADS_PER_BLOCK));
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	point_wise_add_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][a_ind], gpu_buffers[gpu_ind][b_ind], scalar, 
		buffer_sz[gpu_ind][a_ind]/(sizeof(DATA_TYPE)));
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
