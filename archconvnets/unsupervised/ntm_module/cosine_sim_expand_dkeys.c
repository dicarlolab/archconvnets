#define GPU_KEYS gpu_buffers[gpu_ind][keys_ind]
#define GPU_MEM gpu_buffers[gpu_ind][mem_ind]

#define KEYS(A, B) keys[(A)*mem_length + B]
#define MEM(A, B) mem[(A)*mem_length + B]
#define COSED(A, B, C, D) data_out[(A)*M*n_controllers*mem_length + \
	(B)*n_controllers*mem_length + (C)*mem_length + D]

#define MEM_SZ buffer_sz[gpu_ind][mem_ind]
#define KEYS_SZ buffer_sz[gpu_ind][keys_ind]

#define COSED_SZ (n_controllers*M*n_controllers*mem_length*sizeof(DATA_TYPE))
	
__global__ void cosine_sim_expand_dkeys_kernel(float * keys, float * mem, 
			float * data_out, long n_controllers, long mem_length, long M){
	
	int i = blockIdx.x;
	int j = threadIdx.x / mem_length;
	int k = threadIdx.x % mem_length;
	
	float numer = 0, denom, denom2, keys_sq_sum = 0, mem_sq_sum = 0;
	
	
	for(int k_local = 0; k_local < mem_length; k_local++){
		mem_sq_sum += MEM(j,k_local) * MEM(j,k_local);
	}
	mem_sq_sum = sqrt(mem_sq_sum);
	
	/////////////////////////
	// mem*denom - temp (keys*numer*mem_sq_sum)
	for(int k_local = 0; k_local < mem_length; k_local++){
		keys_sq_sum += KEYS(i,k_local) * KEYS(i,k_local);
	}
	keys_sq_sum = sqrt(keys_sq_sum);
	
	denom = keys_sq_sum * mem_sq_sum;
	denom2 = keys_sq_sum * denom * denom / mem_sq_sum;
	for(int k_local = 0; k_local < mem_length; k_local++){
		numer += KEYS(i,k_local) * MEM(j,k_local) / denom2;
	}
	
	COSED(i,j,i,k) = (MEM(j,k) / denom) - (KEYS(i,k) * numer);
	
	for(int i_local = 0; i_local < n_controllers; i_local++){
		if(i_local != i)
			COSED(i,j,i_local,k) = 0;
	}

	return;
}

static PyObject *cosine_sim_expand_dkeys(PyObject *self, PyObject *args){
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
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, COSED_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = COSED_SZ;
	}else if(COSED_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// run kernel
	cosine_sim_expand_dkeys_kernel <<< n_controllers, M*mem_length >>> 
			(GPU_KEYS, GPU_MEM, GPU_BUFFER_OUT, 
			n_controllers, mem_length, M);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
