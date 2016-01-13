__global__ void point_wise_div_sqrt_kernel(float * out, float * a, float * b, float clip, int data_out_numel){
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
		
		out[ind_g] = a[ind_g] / sqrtf(b[ind_g]);
		
		if(out[ind_g] > clip)
			out[ind_g] = clip;
		else if(out[ind_g] < -clip)
			out[ind_g] = -clip;
		
		if(isnan(out[ind_g]) || isinf(out[ind_g]))
			out[ind_g] = 0;
	}
}

static PyObject * point_wise_div_sqrt(PyObject *self, PyObject *args){
	cudaError_t err;
	float clip;
	int a_ind, b_ind, gpu_ind, out_buffer_ind;
	char buffer_prev_init = 1;
	
	if (!PyArg_ParseTuple(args, "iiifi", &a_ind, &b_ind, &out_buffer_ind, &clip, &gpu_ind)) 
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
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][b_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][b_ind];
		buffer_prev_init = 0;
	}else if(buffer_sz[gpu_ind][b_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][a_ind] != buffer_sz[gpu_ind][b_ind]){
		printf("buffer sizes are not equal %li, %li\n", buffer_sz[gpu_ind][a_ind], buffer_sz[gpu_ind][b_ind]);
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(a_ind == out_buffer_ind && buffer_prev_init == 0){
		cudaMemcpy(gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][b_ind], OUT_BUFFER_SZ, cudaMemcpyDeviceToDevice);
	}else{
		// determine number of blocks
		int n_blocks = (int)ceil((double)buffer_sz[gpu_ind][a_ind]/(sizeof(DATA_TYPE)*MAX_THREADS_PER_BLOCK));
		if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
		
		// run kernel
		point_wise_div_sqrt_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][a_ind], gpu_buffers[gpu_ind][b_ind],
			clip, buffer_sz[gpu_ind][a_ind]/(sizeof(DATA_TYPE)));
	}
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
