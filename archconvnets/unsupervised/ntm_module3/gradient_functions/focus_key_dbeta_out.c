#define DFKB_SZ (dim_above*n_imgs*n_controllers*sizeof(DATA_TYPE))
#define KEYS_SZ buffer_sz[gpu_ind][keys_ind]

__global__ void focus_key_dbeta_out_kernel(float * keys, float * deriv_above, float * out_data, int n_controllers, int mem_length, int dim_above){ 
	int img = blockIdx.x / dim_above;
	int a = blockIdx.x % dim_above;
	int i = threadIdx.x;
	
	int ind = img*dim_above*n_controllers + a*n_controllers + i;
	out_data[ind] = 0;
	
	int ind_temp = img*dim_above*n_controllers*mem_length + a*n_controllers*mem_length + i*mem_length;
	
	for(int j = 0; j < mem_length; j++){
		//out_data[a,i] += deriv_above[a,i,j] * KEYS(i,j);
		out_data[ind] += deriv_above[ind_temp + j] * KEYS(img,i,j);
	}
}

// deriv_above: [a, n_imgs, n_controllers, mem_length]
// key: [n_imgs, n_controllers, mem_length]
// beta_out: [n_imgs, n_controllers]

// deriv_above * dfkb = [a, n_imgs, n_controllers] (sum mem_length)

static PyObject * focus_key_dbeta_out(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *keys_shape;
	int keys_ind, out_buffer_ind, gpu_ind, deriv_above_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iii", &keys_ind, &PyTuple_Type, &keys_shape, &deriv_above_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(keys_ind >= N_BUFFERS || keys_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
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
	
	long dim_above = buffer_sz[gpu_ind][deriv_above_ind] / buffer_sz[gpu_ind][keys_ind];
	
	if(n_imgs*n_controllers*mem_length*sizeof(DATA_TYPE) != KEYS_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DFKB_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DFKB_SZ;
	}else if(DFKB_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	focus_key_dbeta_out_kernel <<< dim_above*n_imgs, n_controllers >>> (gpu_buffers[gpu_ind][keys_ind], 
		gpu_buffers[gpu_ind][deriv_above_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], n_controllers, mem_length, dim_above);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	
	Py_INCREF(Py_None);
	return Py_None;
}
