#define COS(A, B) data_out[(A)*M + B]
#define COS_SZ (n_controllers*M*sizeof(DATA_TYPE))

#define KEYS_IND(A, B) ((A)*mem_length + B)
#define MEM_IND(A, B) ((A)*mem_length + B)
#define DATA_OUT_COS(A, B) data_out[(A)*M + B]
	
__global__ void cosine_sim_kernel(float * keys, float * mem, 
			float * data_out, long n_controllers, long mem_length, long M){
	int i = blockIdx.x;
	int j = threadIdx.x;
	
	int k, keys_ind, mem_ind;
	float keys_sq_sum = 0, mem_sq_sum = 0, numer = 0;
	
	keys_ind = KEYS_IND(i,0);
	mem_ind = MEM_IND(j,0);
	for(k = 0; k < mem_length; k++){
		mem_sq_sum += mem[mem_ind] * mem[mem_ind];
		keys_sq_sum += keys[keys_ind] * keys[keys_ind];
		numer += keys[keys_ind] * mem[mem_ind];
		
		keys_ind ++;
		mem_ind ++;
	}
	
	DATA_OUT_COS(i,j) = numer / (sqrt(mem_sq_sum)*sqrt(keys_sq_sum));
}

/*def cosine_sim_expand_reorg(args):
	assert len(args) == 2
	keys, mem = args
	
	n_controllers, mem_length = keys.shape
	M = mem.shape[0]
	
	denom = np.zeros((n_controllers, M))

	for i in range(n_controllers):
		for j in range(M):
			keys_sq_sum = 0
			mem_sq_sum = 0
			numer = 0
			
			for k in range(mem_length):
				mem_sq_sum += mem[j,k]**2
				keys_sq_sum += keys[i,k]**2
				numer += keys[i,k] * mem[j,k]
			mem_sq_sum = np.sqrt(mem_sq_sum)
			keys_sq_sum = np.sqrt(keys_sq_sum)
			
			denom[i,j] = numer / (keys_sq_sum*mem_sq_sum)
	
	return denom # [n_controllers, n_mem_slots]*/

static PyObject *cosine_sim(PyObject *self, PyObject *args){
	PyTupleObject *keys_shape, *mem_shape;
	int keys_ind, mem_ind, out_buffer_ind, gpu_ind;
	cudaError_t err;
	
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &keys_ind, &PyTuple_Type, &keys_shape, 
			&mem_ind, &PyTuple_Type, &mem_shape, &out_buffer_ind, &gpu_ind))
		return NULL;
	
	if(keys_ind >= N_BUFFERS || keys_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || 
			mem_ind >= N_BUFFERS || mem_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(MEM_SZ == 0 || KEYS_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long n_controllers = PyLong_AsLong(PyTuple_GetItem((PyObject *)keys_shape,0));
	long mem_length = PyLong_AsLong(PyTuple_GetItem((PyObject *)keys_shape,1));
	long M = PyLong_AsLong(PyTuple_GetItem((PyObject *)mem_shape,0));
	
	if(n_controllers*mem_length*sizeof(DATA_TYPE) != KEYS_SZ || M*mem_length*sizeof(DATA_TYPE) != MEM_SZ){
		printf("specified input sizes do not equal to stored gpu buffer. dot_cpu()\n");
		return NULL;
	}
	
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, COS_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = COS_SZ;
	}else if(COS_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// run kernel
	cosine_sim_kernel <<< n_controllers, M >>> (GPU_KEYS, GPU_MEM, GPU_BUFFER_OUT, n_controllers, mem_length, M);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
