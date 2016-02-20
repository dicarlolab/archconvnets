#define FOCUS_SZ (n_imgs*n_controllers*mem_length*sizeof(DATA_TYPE))
#define KEYS_SZ buffer_sz[gpu_ind][keys_ind]
#define BETA_OUT_SZ buffer_sz[gpu_ind][beta_out_ind]

__global__ void focus_key_kernel(float * keys, float * beta_out, float * out, int n_controllers, int mem_length, int n_imgs){ 
	int img = blockIdx.x;
	int c = threadIdx.x / mem_length;
	int loc = threadIdx.x % mem_length;
	
	int ind = img*n_controllers*mem_length + c*mem_length + loc;

	//out[img,c,loc] = keys[img,c,loc] * beta_out[img, c];
	out[ind] = keys[ind] * beta_out[img*n_controllers + c];
}

static PyObject * focus_key(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *keys_shape;
	int keys_ind, out_buffer_ind, gpu_ind, beta_out_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iii", &keys_ind, &PyTuple_Type, &keys_shape, &beta_out_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(keys_ind >= N_BUFFERS || keys_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			beta_out_ind >= N_BUFFERS || beta_out_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(keys_shape, 0));
	long n_controllers = PyLong_AsLong(PyTuple_GetItem(keys_shape, 1));
	long mem_length = PyLong_AsLong(PyTuple_GetItem(keys_shape, 2));
	
	if(n_imgs*n_controllers*mem_length*sizeof(DATA_TYPE) != KEYS_SZ || n_imgs*n_controllers*sizeof(DATA_TYPE) != BETA_OUT_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, FOCUS_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = FOCUS_SZ;
	}else if(FOCUS_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	focus_key_kernel <<< n_imgs, n_controllers * mem_length >>> (gpu_buffers[gpu_ind][keys_ind], gpu_buffers[gpu_ind][beta_out_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], n_controllers, mem_length, n_imgs);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
