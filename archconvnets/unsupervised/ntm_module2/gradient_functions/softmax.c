__global__ void softmax_kernel(float * layer_in, float * out, int dim0, int dim1){ 
	int i = blockIdx.x;
	int j = threadIdx.x;

	int ind = i*dim1 + j;
	float exp_layer_in = __expf(layer_in[ind]);
	
	extern __shared__ float shared_mem[];

	float* local_sum = (float*)&shared_mem;
	
	if(j == 0)
		local_sum[0] = 0;
	__syncthreads();
	
	atomicAdd(&local_sum[0], exp_layer_in);
	__syncthreads();
	
	out[ind] = exp_layer_in / local_sum[0];
}

/* exp_layer_in = np.exp(layer_in)
   out = exp_layer_in/np.sum(exp_layer_in,1)[:,np.newaxis]*/

static PyObject *softmax(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *layer_in_shape;
	int gpu_ind, layer_in_ind, out_buffer_ind;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &layer_in_ind, &PyTuple_Type, &layer_in_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(layer_in_ind >= N_BUFFERS || layer_in_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)layer_in_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)layer_in_shape,1));
	
	if(dim0*dim1*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][layer_in_ind]){
		printf("specified input sizes do not equal to stored gpu buffer.\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][layer_in_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][layer_in_ind];
	}else if(buffer_sz[gpu_ind][layer_in_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	softmax_kernel <<< dim0, dim1, sizeof(float) >>> (gpu_buffers[gpu_ind][layer_in_ind], gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
