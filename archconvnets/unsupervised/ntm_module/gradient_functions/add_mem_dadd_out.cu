#define ADD_MEM_DADD_OUT_NUMEL (M * mem_length * C * mem_length)
#define ADD_MEM_DADD_OUT_SZ (ADD_MEM_DADD_OUT_NUMEL*sizeof(DATA_TYPE))

__global__ void add_mem_dadd_out_kernel(float * gw, float * data_out, int mem_length, int M, int C_mem_length, int mem_length_C_mem_length, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, i,j,k,l, remainder;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		// we are computing the output data_out[i,j,k,l]... determine indices
		i = ind_g / mem_length_C_mem_length;
		remainder = ind_g % mem_length_C_mem_length;
		
		j = remainder / C_mem_length;
		remainder = remainder % C_mem_length;
		
		k = remainder / mem_length;
		l = remainder % mem_length;
		
		if(j != l)
			data_out[ind_g] = 0;
		else
			data_out[ind_g] = gw[k*M + i];
	}
}

/*def add_mem_dadd_out(gw):
	temp = np.zeros((M, mem_length, C, mem_length),dtype='single')
	temp[:,range(mem_length),:,range(mem_length)] = gw.T
	return temp*/

// gw = (16, 6)  add_out = (16, 8)
// C, M    ....            C, mem_length

static PyObject *add_mem_dadd_out(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, gw_ind, out_buffer_ind;
	PyObject *gw_shape, *add_out_shape;
	
	if (!PyArg_ParseTuple(args, "iO!O!ii", &gw_ind, &PyTuple_Type, &gw_shape, &PyTuple_Type, &add_out_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || gw_ind >= N_BUFFERS || gw_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long C = PyLong_AsLong(PyTuple_GetItem(gw_shape,0));
	long M = PyLong_AsLong(PyTuple_GetItem(gw_shape,1));
	
	long C2 = PyLong_AsLong(PyTuple_GetItem(add_out_shape,0));
	long mem_length = PyLong_AsLong(PyTuple_GetItem(add_out_shape,1));
	
	if(C != C2){
		printf("inner dot product dimensions do not match\n");
		return NULL;
	}
	
	if(C*M*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][gw_ind]){
		printf("specified input sizes do not equal to stored gpu buffer. dot_cpu()\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, ADD_MEM_DADD_OUT_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = ADD_MEM_DADD_OUT_SZ;
	}else if(ADD_MEM_DADD_OUT_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)ADD_MEM_DADD_OUT_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	add_mem_dadd_out_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][gw_ind], 
			GPU_BUFFER_OUT, mem_length, M, C*mem_length, mem_length*C*mem_length, ADD_MEM_DADD_OUT_NUMEL);
		
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
