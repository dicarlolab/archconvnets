#define DFKK(A, B, C, D) dfkk[(A)*mem_length*n_controllers*mem_length + (B)*n_controllers*mem_length + (C)*mem_length + D]
#define DFKK_SZ (n_controllers*mem_length*n_controllers*mem_length*sizeof(DATA_TYPE))
#define BETA_OUT_SZ buffer_sz[gpu_ind][beta_out_ind]

__global__ void focus_key_dkeys_kernel(float * beta_out, float * dfkk, int n_controllers, int mem_length){ 
	int i = threadIdx.x / mem_length;
	int j = threadIdx.x % mem_length;

	for(int i_local = 0; i_local < n_controllers; i_local++){
		for(int j_local = 0; j_local < mem_length; j_local++){
			DFKK(i,j,i_local,j_local) = 0;
		}
	}
	
	DFKK(i,j,i,j) = beta_out[i];
	
	return;
}

static PyObject * focus_key_dkeys(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *keys_shape;
	int beta_out_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &beta_out_ind, &PyTuple_Type, &keys_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(beta_out_ind >= N_BUFFERS || beta_out_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(BETA_OUT_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long n_controllers = PyLong_AsLong(PyTuple_GetItem((PyObject *)keys_shape,0));
	long mem_length = PyLong_AsLong(PyTuple_GetItem((PyObject *)keys_shape,1));
	
	if(n_controllers*sizeof(DATA_TYPE) != BETA_OUT_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DFKK_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DFKK_SZ;
	}else if(DFKK_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	focus_key_dkeys_kernel <<< 1, n_controllers * mem_length >>> (gpu_buffers[gpu_ind][beta_out_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], n_controllers, mem_length);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
