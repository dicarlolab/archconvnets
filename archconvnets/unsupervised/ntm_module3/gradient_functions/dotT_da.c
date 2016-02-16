#define ADD_MEM_DGW_NUMEL (dim_above * C * M)
#define ADD_MEM_DGW_SZ (n_imgs*ADD_MEM_DGW_NUMEL*sizeof(DATA_TYPE))

__global__ void add_mem_dgw_kernel(float * add_out, float * deriv_above, float * data_out, int mem_length, int M, int C, 
		int dim_above, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, remainder, mem, ind_temp, a, c, m;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		// we are computing the output data_out[a, C, M]... determine indices
		a = ind_g / (C*M);
		remainder = ind_g % (C*M);
		
		c = remainder / M;
		m = remainder % M;
		
		data_out[ind_g] = 0;
		ind_temp = a*M*mem_length + m*mem_length;
		for(mem = 0; mem < mem_length; mem++){
			//data_out[ind_g] += deriv_above[a, m, mem] * add_out[c, mem];
			data_out[ind_g] += deriv_above[ind_temp + mem] * add_out[c*mem_length + mem];
		}
	}
}

/*def add_mem_dgw(add_out):
	temp = np.zeros((M, mem_length, C, M),dtype='single')
	temp[range(M),:,:,range(M)] = add_out.T
	return temp*/

// gw = (16, 6)  add_out = (16, 8)
// C, M    ....            C, mem_length

// deriv_above = (a, M, mem_length)
// deriv_above (a, M, mem_length) * dotT_da (M, mem_length, C, M) = [a, C, M]


static PyObject *dotT_da(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, add_out_ind, out_buffer_ind, deriv_above_ind, n_imgs;
	PyObject *gw_shape, *add_out_shape, *deriv_above_shape;
	
	if (!PyArg_ParseTuple(args, "iO!O!iO!iii", &add_out_ind, &PyTuple_Type, &gw_shape, &PyTuple_Type, &add_out_shape, &deriv_above_ind, 
		&PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &n_imgs, &gpu_ind)) 
		return NULL;
        
	if(out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || add_out_ind >= N_BUFFERS || add_out_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	int dim_offset = 0;
	if(n_imgs > 1)
		dim_offset ++;
	
	// get sizes
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape, 0));
	
	long C = PyLong_AsLong(PyTuple_GetItem(gw_shape, dim_offset));
	long M = PyLong_AsLong(PyTuple_GetItem(gw_shape, 1 + dim_offset));
	
	long C2 = PyLong_AsLong(PyTuple_GetItem(add_out_shape, dim_offset));
	long mem_length = PyLong_AsLong(PyTuple_GetItem(add_out_shape, 1 + dim_offset));
	
	if(C != C2){
		printf("inner dot product dimensions do not match\n");
		return NULL;
	}
	
	if(n_imgs*C*mem_length*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][add_out_ind]){
		printf("specified input sizes do not equal to stored gpu buffer. %s\n", __FILE__);
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, ADD_MEM_DGW_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = ADD_MEM_DGW_SZ;
	}else if(ADD_MEM_DGW_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)ADD_MEM_DGW_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	
	for(int batch = 0; batch < n_imgs; batch++){
		// run kernel
		add_mem_dgw_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][add_out_ind] + batch*C*mem_length, 
				gpu_buffers[gpu_ind][deriv_above_ind] + batch*M*mem_length, 
				GPU_BUFFER_OUT + batch*ADD_MEM_DGW_NUMEL, mem_length, M, C, dim_above, ADD_MEM_DGW_NUMEL);
	}
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
