#define N_SHIFTS 3
#define DSHDW_SZ (dim_above*C*M*sizeof(DATA_TYPE))
#define SHIFT_OUT(A, B) shift_out[(A)*N_SHIFTS + B]

__global__ void shift_w_dw_interp_kernel(float * shift_out, float * deriv_above, float * data_out, int C, int M){ 
	int a = blockIdx.x;
	int c = threadIdx.x / N_SHIFTS;
	int loc_ind = threadIdx.x % N_SHIFTS; // 0,1,2
	
	for(int loc = 0; loc < M; loc++){
		int loc_dshdw_ind = loc + loc_ind - 1;
		if(loc_dshdw_ind == -1)
			loc_dshdw_ind = M - 1;
		else if(loc_dshdw_ind == M)
			loc_dshdw_ind = 0;
		
		//DSHDW(c, loc, loc_dshdw_ind) = SHIFT_OUT(c, loc_ind);
		//atomicAdd(&data_out[a, c, loc_dshdw_ind], SHIFT_OUT(c, loc_ind) * deriv_above[a, c, loc]);
		atomicAdd(&data_out[a*C*M + c*M + loc_dshdw_ind], SHIFT_OUT(c, loc_ind) * deriv_above[a*C*M + c*M + loc]);
	}
}

// shift_out: [C, n_shifts], w_interp: [C, M]
// deriv_above: [dim_above, C, M]
//
// dsdw: [C, M, C, M] --> [C, M, M]

// deriv_above * dsds = [dim_above, C, M] (sum across first M)


static PyObject * shift_w_dw_interp(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *w_interp_shape, *deriv_above_shape;
	int shift_out_ind, out_buffer_ind, gpu_ind, deriv_above_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &shift_out_ind, &PyTuple_Type, &w_interp_shape, &deriv_above_ind,
		&PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &gpu_ind)) 
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
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0));
	long C = PyLong_AsLong(PyTuple_GetItem(w_interp_shape,0));
	long M = PyLong_AsLong(PyTuple_GetItem(w_interp_shape,1));
	
	if(C*N_SHIFTS*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][shift_out_ind]){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DSHDW_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DSHDW_SZ;
	}else if(DSHDW_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}

	cudaMemset(gpu_buffers[gpu_ind][out_buffer_ind], 0, OUT_BUFFER_SZ); CHECK_CUDA_ERR
	
	shift_w_dw_interp_kernel <<< dim_above, C * N_SHIFTS >>> (gpu_buffers[gpu_ind][shift_out_ind], 
		gpu_buffers[gpu_ind][deriv_above_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], C, M);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
