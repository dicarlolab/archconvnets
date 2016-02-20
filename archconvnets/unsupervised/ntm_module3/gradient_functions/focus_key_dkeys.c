#define DFKK_SZ (n_imgs*dim_above*n_controllers*mem_length*sizeof(DATA_TYPE))
#define BETA_OUT_SZ buffer_sz[gpu_ind][beta_out_ind]

__global__ void focus_key_dkeys_kernel(float * beta_out, float * deriv_above, float * out_data, 
		int n_controllers, int mem_length, int dim_above){ 
	int img = blockIdx.x / dim_above;
	int dim = blockIdx.x % dim_above;
	int c = threadIdx.x / mem_length;
	int loc = threadIdx.x % mem_length;

	int ind = img*dim_above*n_controllers*mem_length + dim*n_controllers*mem_length + c*mem_length + loc;
	out_data[ind] = beta_out[img*n_controllers + c] * deriv_above[ind];
}

// deriv_above: [a, n_controllers, mem_length]
// key: [n_controllers, mem_length]
// beta_out: [n_controllers]
// dfkk: [n_controllers, mem_length, n_controllers, mem_length] -> [n_controllers, mem_length]

// deriv_above * dfkb = [a, n_controllers, mem_length] (sum mem_length)


static PyObject * focus_key_dkeys(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *keys_shape;
	int beta_out_ind, out_buffer_ind, gpu_ind, deriv_above_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iii", &beta_out_ind, &PyTuple_Type, &keys_shape, &deriv_above_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(beta_out_ind >= N_BUFFERS || beta_out_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
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
	
	long dim_above = buffer_sz[gpu_ind][deriv_above_ind] / (n_imgs*n_controllers*mem_length*sizeof(DATA_TYPE));
	
	if(n_imgs*n_controllers*sizeof(DATA_TYPE) != BETA_OUT_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DFKK_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DFKK_SZ;
	}else if(DFKK_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	focus_key_dkeys_kernel <<< dim_above*n_imgs, n_controllers * mem_length >>> (gpu_buffers[gpu_ind][beta_out_ind], 
		gpu_buffers[gpu_ind][deriv_above_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], n_controllers, mem_length, dim_above);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
