#define DSDL(A, B, C, D) dsdl[(A)*dim1*dim0*dim1 + (B)*dim0*dim1 + (C)*dim1 + D]
#define LAYER_OUT_SM(A, B) layer_out[(A)*dim1 + (B)]
#define DSDL_SZ (dim0*dim1*dim0*sizeof(DATA_TYPE))
#define LAYER_OUT_SZ buffer_sz[gpu_ind][layer_out_ind]

__global__ void sigmoid_dlayer_in_kernel(float * layer_out, float * dsdl, int dim0, int dim1){ 
	int i = threadIdx.x / dim1;
	int j = threadIdx.x % dim1;

	DSDL(i,j,i,j) = LAYER_OUT_SM(i,j) * (1 - LAYER_OUT_SM(i,j));
	
	for(int i_local = 0; i_local < dim0; i_local++){
		for(int j_local = 0; j_local < dim1; j_local++){
			if(i_local != i || j_local != j)
				DSDL(i,j,i_local,j_local) = 0;
		}
	}

	return;
}

static PyObject * sigmoid_dlayer_in(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *layer_out_shape;
	int layer_out_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &layer_out_ind, &PyTuple_Type, &layer_out_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(layer_out_ind >= N_BUFFERS || layer_out_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(LAYER_OUT_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)layer_out_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)layer_out_shape,1));
	
	if(dim0*dim1*sizeof(DATA_TYPE) != LAYER_OUT_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DSDL_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DSDL_SZ;
	}else if(DSDL_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	sigmoid_dlayer_in_kernel <<< 1, dim0 * dim1 >>> (gpu_buffers[gpu_ind][layer_out_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
