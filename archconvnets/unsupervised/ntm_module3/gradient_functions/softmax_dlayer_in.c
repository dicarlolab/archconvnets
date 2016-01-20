#define LAYER_OUT(A, B) layer_out[(A)*dim1 + B]
#define LAYER_OUT_SZ buffer_sz[gpu_ind][layer_out_ind]
#define SMDLAYER(A, B, C, D) smdlayer[(A)*dim1*dim0*dim1 + \
	(B)*dim0*dim1 + (C)*dim1 + D]
#define SMDLAYER_SZ (dim0*dim1*dim0*dim1*sizeof(DATA_TYPE))

__global__ void softmax_dlayer_in_kernel(float * layer_out, float * smdlayer, long dim0, long dim1){
	int i = blockIdx.x;
	int j = threadIdx.x / dim1;
	int k = threadIdx.x % dim1;

	if(j == k)
		SMDLAYER(i,j,i,j) = LAYER_OUT(i,j) * (1 - LAYER_OUT(i,j));
	else
		SMDLAYER(i,j,i,k) = -LAYER_OUT(i,j)*LAYER_OUT(i,k);
	
	for(int i_local = 0; i_local < dim0; i_local++){
		if(i_local != i)
			SMDLAYER(i,j,i_local,k) = 0;
	}

	return;
}

static PyObject *softmax_dlayer_in(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *layer_out_shape;
	int layer_out_ind, gpu_ind, out_buffer_ind;

	if (!PyArg_ParseTuple(args, "iO!ii", &layer_out_ind, &PyTuple_Type, &layer_out_shape,
			&out_buffer_ind, &gpu_ind)) 
		return NULL;
		
	if(layer_out_ind >= N_BUFFERS || layer_out_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
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

	long dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)layer_out_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)layer_out_shape,1));

	if(dim0*dim1*sizeof(DATA_TYPE) != LAYER_OUT_SZ){
		printf("specified input sizes do not equal stored gpu buffer. softmax_dlayer_in()\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, SMDLAYER_SZ); MALLOC_ERR_CHECK

		OUT_BUFFER_SZ = SMDLAYER_SZ;
	}else if(SMDLAYER_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}

	softmax_dlayer_in_kernel <<< dim0, dim1*dim1 >>> (gpu_buffers[gpu_ind][layer_out_ind], GPU_BUFFER_OUT, dim0, dim1);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
