#define SIGMOID_DLAYER_IN_NUMEL (buffer_sz[gpu_ind][layer_out_ind] / sizeof(DATA_TYPE))
#define SIGMOID_DLAYER_IN_BUFFER_SZ (buffer_sz[gpu_ind][layer_out_ind]*buffer_sz[gpu_ind][layer_out_ind]/sizeof(DATA_TYPE))

// we only have the kernel run for dim1*dim2, even though the output is dim1*dim2*dim1*dim2 because i=k and j=l, everything else is zero
__global__ void sigmoid_dlayer_in_kernel(float * layer_out, float * out_buffer, int layer_out_dim2, 
		int layer_out_dim2_layer_out_dim1_layer_out_dim2, int layer_out_dim1_layer_out_dim2, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, ind_g_out,i,j;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		// we are computing the output data_out[i,j,i,j]... determine indices into layer_out
		
		i = ind_g / layer_out_dim2;
		j = ind_g % layer_out_dim2;
		
		ind_g_out = i*layer_out_dim2_layer_out_dim1_layer_out_dim2 + j*layer_out_dim1_layer_out_dim2 + ind_g;
			
		out_buffer[ind_g_out] = layer_out[ind_g] * (1-layer_out[ind_g]);
	}
}

static PyObject *sigmoid_dlayer_in(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, layer_out_ind, out_buffer_ind;
	PyObject *layer_out_shape;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &layer_out_ind, &PyTuple_Type, &layer_out_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(layer_out_ind >= N_BUFFERS || layer_out_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long layer_out_dim1 = PyLong_AsLong(PyTuple_GetItem(layer_out_shape,0));
	long layer_out_dim2 = PyLong_AsLong(PyTuple_GetItem(layer_out_shape,1));
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(layer_out_dim1*layer_out_dim2*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][layer_out_ind]){
		printf("specified input sizes do not equal to stored gpu buffer.\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, SIGMOID_DLAYER_IN_BUFFER_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = SIGMOID_DLAYER_IN_BUFFER_SZ;
	}else if(SIGMOID_DLAYER_IN_BUFFER_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	err = cudaMemset(GPU_BUFFER_OUT, 0, SIGMOID_DLAYER_IN_BUFFER_SZ);  MALLOC_ERR_CHECK
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)SIGMOID_DLAYER_IN_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	sigmoid_dlayer_in_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][layer_out_ind], gpu_buffers[gpu_ind][out_buffer_ind], 
		layer_out_dim2, layer_out_dim2 * layer_out_dim1 * layer_out_dim2, layer_out_dim1 * layer_out_dim2, SIGMOID_DLAYER_IN_NUMEL);
		
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
