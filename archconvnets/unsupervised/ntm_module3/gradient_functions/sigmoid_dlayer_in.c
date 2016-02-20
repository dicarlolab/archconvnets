#define SIGMOID_DLAYER_IN_NUMEL (buffer_sz[gpu_ind][deriv_above_ind] / sizeof(DATA_TYPE))
#define SIGMOID_DLAYER_IN_SZ buffer_sz[gpu_ind][deriv_above_ind]

__global__ void sigmoid_dlayer_in_kernel(float * layer_out, float * deriv_above, float * out_buffer, 
		int dim_above, int layer_sz, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, ind_g_local, img, loc;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		img = ind_g / (dim_above*layer_sz);
		//r = ind_g % (dim_above*layer_sz);
		//loc = r % layer_sz;
		loc = (ind_g % (dim_above*layer_sz)) % layer_sz;
		
		ind_g_local = img*layer_sz + loc;
		
		out_buffer[ind_g] = deriv_above[ind_g] * layer_out[ind_g_local] * (1-layer_out[ind_g_local]);
	}
}

static PyObject *sigmoid_dlayer_in(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, layer_out_ind, deriv_above_ind, out_buffer_ind;
	PyObject *deriv_above_shape;
	
	if (!PyArg_ParseTuple(args, "iiO!ii", &layer_out_ind, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &gpu_ind)) 
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
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0));
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,1));
	long layer_sz = buffer_sz[gpu_ind][deriv_above_ind] / (n_imgs * dim_above * sizeof(DATA_TYPE));
	
	if(n_imgs*layer_sz != buffer_sz[gpu_ind][layer_out_ind]/sizeof(DATA_TYPE)){
		printf("input dims don't match %s %i... n_imgs %li, layer_sz %li, layer_out_sz: %li\n", __FILE__,__LINE__,
			n_imgs, layer_sz, buffer_sz[gpu_ind][layer_out_ind]/sizeof(DATA_TYPE));
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, SIGMOID_DLAYER_IN_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = SIGMOID_DLAYER_IN_SZ;
	}else if(SIGMOID_DLAYER_IN_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)SIGMOID_DLAYER_IN_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	sigmoid_dlayer_in_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][layer_out_ind], 
		gpu_buffers[gpu_ind][deriv_above_ind], gpu_buffers[gpu_ind][out_buffer_ind], dim_above, layer_sz, SIGMOID_DLAYER_IN_NUMEL);
		
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
