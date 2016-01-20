#define RELU_NUMEL (buffer_sz[gpu_ind][layer_in_ind] / sizeof(DATA_TYPE))

__global__ void relu_kernel(float * layer_in, float * out_buffer, int data_out_numel){
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
		
		// we are computing the output data_out[i,j]... determine start indices of data1 & data2 for summation:
		
		if(layer_in[ind_g] < 0)
			out_buffer[ind_g] = 0;
		else
			out_buffer[ind_g] = layer_in[ind_g];
	}
}

static PyObject *relu(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, layer_in_ind, out_buffer_ind;
	
	if (!PyArg_ParseTuple(args, "iii", &layer_in_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(layer_in_ind >= N_BUFFERS || layer_in_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][layer_in_ind] <= 0){
		printf("input buffer not initialized\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][layer_in_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][layer_in_ind];
	}else if(buffer_sz[gpu_ind][layer_in_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)SIGMOID_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	relu_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][layer_in_ind], gpu_buffers[gpu_ind][out_buffer_ind], RELU_NUMEL);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
