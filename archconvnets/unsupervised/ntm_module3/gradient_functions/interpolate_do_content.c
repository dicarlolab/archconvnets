__global__ void interpolate_do_content_kernel(float * interp_gate_out, float * dido, int dim0, int dim1){ 
	int i = threadIdx.x / dim1;
	int j = threadIdx.x % dim1;

	for(int i_local = 0; i_local < dim0; i_local++){
		for(int j_local = 0; j_local < dim1; j_local++){
			DIDO(i,j,i_local,j_local) = 0;
	}}

	DIDO(i,j,i,j) = interp_gate_out[i];//INTERP_GATE_OUT(i,j);

	return;
}

static PyObject * interpolate_do_content(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *o_prev_shape;
	int interp_gate_out_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &interp_gate_out_ind, &PyTuple_Type, &o_prev_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(interp_gate_out_ind >= N_BUFFERS || interp_gate_out_ind < 0 ||  out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)o_prev_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)o_prev_shape,1));
	
	if(dim0*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][interp_gate_out_ind]){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DIDO_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DIDO_SZ;
	}else if(DIDO_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	interpolate_do_content_kernel <<< 1, dim0*dim1 >>> (gpu_buffers[gpu_ind][interp_gate_out_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
