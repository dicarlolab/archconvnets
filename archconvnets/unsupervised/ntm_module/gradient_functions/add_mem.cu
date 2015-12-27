#define ADD_MEM_OUT_NUMEL (gw_dim2*add_out_dim2)
#define ADD_MEM_OUT_SZ (ADD_MEM_OUT_NUMEL*sizeof(DATA_TYPE))

__global__ void add_mem_kernel(float * gw, float * add_out, float * data_out, int gw_dim2, int add_out_dim1, 
			int add_out_dim2, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, gw_ind, add_out_ind;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		// we are computing the output data_out[i,j]... determine start indices of gw & add_out for summation:
		
		gw_ind = ind_g / add_out_dim2; // i = ind_g / add_out_dim2;   gw_ind = DATA1_IND(0,i);
		add_out_ind = ind_g % add_out_dim2; //   j = ind_g % add_out_dim2;   ADD_OUT_IND(0,j);
		
		data_out[ind_g] = 0;
		for(int k = 0; k < add_out_dim1; k++){
			data_out[ind_g] += gw[gw_ind] * add_out[add_out_ind];
			
			gw_ind += gw_dim2;
			add_out_ind += add_out_dim2;
		}
	}
}

// def add_mem(gw, add_out):
//	return np.dot(gw.T, add_out)

static PyObject *add_mem(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, gw_ind, add_out_ind, out_buffer_ind;
	PyObject *gw_shape, *add_out_shape;
	
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &gw_ind, &PyTuple_Type, &gw_shape, &add_out_ind, 
			&PyTuple_Type, &add_out_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(gw_ind >= N_BUFFERS || gw_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || 
			add_out_ind >= N_BUFFERS || add_out_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long gw_dim1 = PyLong_AsLong(PyTuple_GetItem(gw_shape,0));
	long gw_dim2 = PyLong_AsLong(PyTuple_GetItem(gw_shape,1));
	
	long add_out_dim1 = PyLong_AsLong(PyTuple_GetItem(add_out_shape,0));
	long add_out_dim2 = PyLong_AsLong(PyTuple_GetItem(add_out_shape,1));
	
	if(gw_dim1 != add_out_dim1){
		printf("inner dot product dimensions do not match\n");
		return NULL;
	}
	
	if(gw_dim1*gw_dim2*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][gw_ind] || 
				add_out_dim1*add_out_dim2*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][add_out_ind]){
		printf("specified input sizes do not equal to stored gpu buffer. dot_cpu()\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, ADD_MEM_OUT_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = ADD_MEM_OUT_SZ;
	}else if(ADD_MEM_OUT_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)ADD_MEM_OUT_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	add_mem_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][gw_ind], gpu_buffers[gpu_ind][add_out_ind], 
			GPU_BUFFER_OUT, gw_dim2, add_out_dim1, add_out_dim2, ADD_MEM_OUT_NUMEL);
		
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
