#define N_SHIFTS 3
#define DSHDW(A, B, Ci, D) dshdw[(A)*M*C*M + (B)*C*M + (Ci)*M + D]
#define DSHDW_SZ (C*M*C*M*sizeof(DATA_TYPE))
#define SHIFT_OUT(A, B) shift_out[(A)*N_SHIFTS + B]

__global__ void shift_w_dw_interp_kernel(float * shift_out, float * dshdw, int C, int M){ 
	int c = threadIdx.x / M;
	int loc = threadIdx.x % M;
	int loc_ind = blockIdx.x; // 0,1,2
	
	int loc_dshdw_ind = loc + loc_ind - 1;
	if(loc_dshdw_ind == -1)
		loc_dshdw_ind = M - 1;
	else if(loc_dshdw_ind == M)
		loc_dshdw_ind = 0;
	
	DSHDW(c, loc, c, loc_dshdw_ind) = SHIFT_OUT(c, loc_ind);

	return;
}

static PyObject * shift_w_dw_interp(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *w_interp_shape;
	int shift_out_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &shift_out_ind, &PyTuple_Type, &w_interp_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(shift_out_ind >= N_BUFFERS || shift_out_ind < 0 ||
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long C = PyLong_AsLong(PyTuple_GetItem((PyObject *)w_interp_shape,0));
	long M = PyLong_AsLong(PyTuple_GetItem((PyObject *)w_interp_shape,1));
	
	if(C*N_SHIFTS*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][shift_out_ind]){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DSHDW_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DSHDW_SZ;
	}else if(DSHDW_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR

	cudaMemset(gpu_buffers[gpu_ind][out_buffer_ind], 0, OUT_BUFFER_SZ); CHECK_CUDA_ERR
	
	shift_w_dw_interp_kernel <<< N_SHIFTS, C * M  >>> (gpu_buffers[gpu_ind][shift_out_ind], gpu_buffers[gpu_ind][out_buffer_ind], C, M);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
