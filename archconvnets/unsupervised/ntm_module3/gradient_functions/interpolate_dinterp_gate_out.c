#define DIDG(A, B, C) didg[(A)*dim1*dim0 + (B)*dim0 + (C)]
#define DIDG_SZ (dim0*dim1*dim0*sizeof(DATA_TYPE))
#define O_CONTENT(A, B) o_content[(A)*dim1 + B]
#define O_PREV(A, B) o_prev[(A)*dim1 + B]

__global__ void interpolate_dinterp_gate_out_kernel(float * o_content, float * o_prev, 
		float * didg, int dim0, int dim1){ 
	int i = threadIdx.x / dim1;
	int j = threadIdx.x % dim1;

	DIDG(i,j,i) = O_CONTENT(i,j) - O_PREV(i,j);
	
	for(int i_local = 0; i_local < dim0; i_local++){
		if(i_local != i)
			DIDG(i,j,i_local) = 0;
	}

	return;
}

static PyObject * interpolate_dinterp_gate_out(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *o_content_shape;
	int o_content_ind, o_prev_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iii", &o_content_ind, &PyTuple_Type, &o_content_shape, &o_prev_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(o_content_ind >= N_BUFFERS || o_content_ind < 0 ||
			o_prev_ind >= N_BUFFERS || o_prev_ind < 0 ||
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)o_content_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)o_content_shape,1));
	
	if(dim0*dim1*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][o_content_ind] ||
		dim0*dim1*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][o_prev_ind]){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DIDG_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DIDG_SZ;
	}else if(DIDG_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	interpolate_dinterp_gate_out_kernel <<< 1, dim0*dim1 >>> (gpu_buffers[gpu_ind][o_content_ind], gpu_buffers[gpu_ind][o_prev_ind], gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
