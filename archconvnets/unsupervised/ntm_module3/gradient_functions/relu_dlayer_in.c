#define DRDL(A, B, C, D) drdl[(A)*dim1*dim0*dim1 + (B)*dim0*dim1 + (C)*dim1 + D]
#define LAYER_IN(A, B) layer_in[(A)*dim1 + (B)]
#define DRDL_SZ (dim0*dim1*dim0*dim1*sizeof(DATA_TYPE))
#define LAYER_IN_SZ buffer_sz[gpu_ind][layer_in_ind]

__global__ void relu_dlayer_in_kernel(float * layer_in, float * drdl, int thresh, int dim0, int dim1){ 
	int i = threadIdx.x / dim1;
	int j = threadIdx.x % dim1;

	for(int i_local = 0; i_local < dim0; i_local++){
		for(int j_local = 0; j_local < dim1; j_local++){
			DRDL(i,j,i_local,j_local) = 0;
		}
	}
	
	if(LAYER_IN(i,j) > thresh)
		DRDL(i,j,i,j) = 1;

	return;
}

static PyObject * relu_dlayer_in(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *layer_in_shape;
	int layer_in_ind, out_buffer_ind, gpu_ind, thresh;
	
	if (!PyArg_ParseTuple(args, "iO!iii", &layer_in_ind, &PyTuple_Type, &layer_in_shape, &out_buffer_ind, &thresh, &gpu_ind)) 
		return NULL;
    
	if(layer_in_ind >= N_BUFFERS || layer_in_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(LAYER_IN_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)layer_in_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)layer_in_shape,1));
	
	if(dim0*dim1*sizeof(DATA_TYPE) != LAYER_IN_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DRDL_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DRDL_SZ;
	}else if(DRDL_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	relu_dlayer_in_kernel <<< 1, dim0 * dim1 >>> (gpu_buffers[gpu_ind][layer_in_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], thresh, dim0, dim1);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
