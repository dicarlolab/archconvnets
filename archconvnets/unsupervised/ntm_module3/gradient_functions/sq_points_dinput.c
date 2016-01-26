#define INPUT(A, B) input[(A)*dim1 + B]
#define OUT(A, B, C, D) out[(A)*dim1*dim0*dim1 + (B)*dim0*dim1 + (C)*dim1 + D]
#define SQ_OUT_SZ (dim0*dim1*dim0*dim1*sizeof(DATA_TYPE))

__global__ void sq_points_dinput_kernel(float * input, float * out, long dim0, long dim1){
	int i = blockIdx.x;
	int j = threadIdx.x / dim1;
	int k = threadIdx.x % dim1;

	if(j == k)
		OUT(i,j,i,j) = 2*INPUT(i,j);
	else
		OUT(i,j,i,k) = 0;
	
	for(int i_local = 0; i_local < dim0; i_local++){
		if(i_local != i)
			OUT(i,j,i_local,k) = 0;
	}

	return;
}

static PyObject *sq_points_dinput(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *input_shape;
	int input_ind, gpu_ind, out_buffer_ind;

	if (!PyArg_ParseTuple(args, "iO!ii", &input_ind, &PyTuple_Type, &input_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
		
	if(input_ind >= N_BUFFERS || input_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}

	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	long dim0 = PyLong_AsLong(PyTuple_GetItem(input_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem(input_shape,1));

	if(dim0*dim1*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][input_ind]){
		printf("specified input sizes do not equal stored gpu buffer.\n");
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, SQ_OUT_SZ); MALLOC_ERR_CHECK

		OUT_BUFFER_SZ = SQ_OUT_SZ;
	}else if(OUT_BUFFER_SZ != SQ_OUT_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}

	sq_points_dinput_kernel <<< dim0, dim1*dim1 >>> (gpu_buffers[gpu_ind][input_ind], GPU_BUFFER_OUT, dim0, dim1);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
