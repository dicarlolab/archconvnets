#define ADD_MEM_DADD_OUT_NUMEL (dim_above * C * mem_length)
#define ADD_MEM_DADD_OUT_SZ (ADD_MEM_DADD_OUT_NUMEL*sizeof(DATA_TYPE))

__global__ void add_mem_dadd_out_kernel(float * gw, float * deriv_above, float * data_out, int mem_length, int M, int C, int dim_above, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, remainder, ind_temp, a, c, m, mem;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		// we are computing the output data_out[a, c, mem]... determine indices
		a = ind_g / (C*mem_length);
		remainder = ind_g % (C*mem_length);
		
		c = remainder / mem_length;
		mem = remainder % mem_length;
		
		data_out[ind_g] = 0;
		ind_temp = a*M*mem_length + mem;
		for(m = 0; m < M; m++)
			data_out[ind_g] += deriv_above[ind_temp + m*mem_length] * gw[c*M + m];
			//data_out[ind_g] += deriv_above[a, m, mem] * gw[c, m];
		
	}
}

/*def add_mem_dadd_out(gw):
	temp = np.zeros((M, mem_length, C, mem_length),dtype='single')
	temp[:,range(mem_length),:,range(mem_length)] = gw.T
	return temp*/

// gw = (16, 6)  add_out = (16, 8)
// C, M    ....            C, mem_length

// deriv_above = (a, M, mem_length)
// deriv_above (a, M, mem_length) * dotT_db (M, mem_length, C, mem_length) = [a, C, mem_length]
//  deriv_above (a, M, mem_length) * gw (C, M) = [a, C, mem_length]

static PyObject *dotT_db(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, gw_ind, out_buffer_ind, deriv_above_ind;
	PyObject *gw_shape, *add_out_shape, * deriv_above_shape;
	
	if (!PyArg_ParseTuple(args, "iO!O!iO!ii", &gw_ind, &PyTuple_Type, &gw_shape, &PyTuple_Type, &add_out_shape, 
		&deriv_above_ind, &PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &gpu_ind)) 
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
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0));
	
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
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, ADD_MEM_DADD_OUT_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = ADD_MEM_DADD_OUT_SZ;
	}else if(ADD_MEM_DADD_OUT_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)ADD_MEM_DADD_OUT_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	add_mem_dadd_out_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][gw_ind], gpu_buffers[gpu_ind][deriv_above_ind], 
			GPU_BUFFER_OUT, mem_length, M, C, dim_above, ADD_MEM_DADD_OUT_NUMEL);
		
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
