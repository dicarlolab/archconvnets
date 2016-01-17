#define MULT_POINTS_DINPUT_NUMEL (a_dim0*a_dim1)
#define MULT_POINTS_DINPUT_SZ (a_dim0*a_dim1*a_dim0*a_dim1*sizeof(DATA_TYPE))

__global__ void mult_points_dinput_kernel(float * a, float * out, int a_dim1, int a_dim1_a_dim0_a_dim1, int a_dim0_a_dim1, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, ind_l, i, j;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		i = ind_g / a_dim1;
		j = ind_g % a_dim1;
		
		ind_l = i*a_dim1_a_dim0_a_dim1 + j*a_dim0_a_dim1 + i*a_dim1 + j;
		
		out[ind_l] = a[ind_g];
	}
}

static PyObject * mult_points_dinput(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, out_buffer_ind, a_ind;
	PyObject *a_shape;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &a_ind, &PyTuple_Type, &a_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(out_buffer_ind >= N_BUFFERS){ 
		printf("buffer index incorrect.\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long a_dim0 = PyLong_AsLong(PyTuple_GetItem(a_shape,0));
	long a_dim1 = PyLong_AsLong(PyTuple_GetItem(a_shape,1));

	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, MULT_POINTS_DINPUT_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = MULT_POINTS_DINPUT_SZ;
	}else if(OUT_BUFFER_SZ != MULT_POINTS_DINPUT_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)MULT_POINTS_DINPUT_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	err = cudaMemset(gpu_buffers[gpu_ind][out_buffer_ind], 0, MULT_POINTS_DINPUT_SZ);  MALLOC_ERR_CHECK
	
	// run kernel
	mult_points_dinput_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][a_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], a_dim1, a_dim1*a_dim0*a_dim1, 
		a_dim0*a_dim1, MULT_POINTS_DINPUT_NUMEL);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
