#define N_SHIFTS 3
#define DSDS(A, B, Ci, D) dsds[(A)*M*C*N_SHIFTS + (B)*C*N_SHIFTS + (Ci)*N_SHIFTS + D]
#define DSDS_SZ (C*M*C*N_SHIFTS*sizeof(DATA_TYPE))
#define W_INTERP(A, B) w_interp[(A)*M + B]

__global__ void shift_w_dshift_out_kernel(float * w_interp, 
		float * dsds, int C, int M){ 
	int c = threadIdx.x / M;
	int m = threadIdx.x % M;
	int H = blockIdx.x - 1; // -1,0,1  
	
	int loc = (m+H)%M;
	if(loc == -1)
		loc = M - 1;
	
	DSDS(c,m,c,H+1) = W_INTERP(c, loc);
	
	for(int c_local = 0; c_local < C; c_local++){
		if(c_local != c)
			DSDS(c,m,c_local,H+1) = 0;
	}

	return;
}

static PyObject * shift_w_dshift_out(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *w_interp_shape;
	int w_interp_ind, out_buffer_ind, gpu_ind;
	
	// (w_interp_ind, w_interp_shape, out_buffer_ind, gpu_ind
	if (!PyArg_ParseTuple(args, "iO!ii", &w_interp_ind, &PyTuple_Type, &w_interp_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(w_interp_ind >= N_BUFFERS || w_interp_ind < 0 ||
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
	
	if(C*M*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][w_interp_ind]){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DSDS_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DSDS_SZ;
	}else if(DSDS_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	shift_w_dshift_out_kernel <<< N_SHIFTS, C * M  >>> (
		gpu_buffers[gpu_ind][w_interp_ind],	gpu_buffers[gpu_ind][out_buffer_ind], C, M);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
