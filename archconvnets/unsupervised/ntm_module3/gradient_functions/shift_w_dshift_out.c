#define N_SHIFTS 3
#define W_INTERP(A, B) w_interp[(A)*M + B]

__global__ void shift_w_dshift_out_kernel(float * w_interp, float * deriv_above,
		float * out_data, int C, int M){ 
	int a = blockIdx.x;
	int c = threadIdx.x / N_SHIFTS;
	int H = (threadIdx.x % N_SHIFTS) - 1; // -1,0,1  
	
	int loc;
	
	int ind = a*C*N_SHIFTS + c*N_SHIFTS + H+1; // [a, c, H+1]
	
	out_data[ind] = 0;
	
	int ind_temp = a*C*M + c*M; //deriv_above[a,c,m];
	
	for(int m = 0; m < M; m++){
		loc = (m+H)%M;
		if(loc == -1)
			loc = M - 1;
		
		//DSDS(c,m,H+1) = W_INTERP(c, loc);
		//out_data[ind] += W_INTERP(c, loc) * deriv_above[a,c,m];
		out_data[ind] += W_INTERP(c, loc) * deriv_above[ind_temp + m];
	}
}

// shift_out: [C, n_shifts], w_interp: [C, M]
// deriv_above: [dim_above, C, M]
//
// dsds: [C, M, C, n_shifts] --> [C, M, n_shifts]

// deriv_above * dsds = [dim_above, C, n_shifts] (sum across M)

static PyObject * shift_w_dshift_out(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *w_interp_shape, *deriv_above_shape;
	int w_interp_ind, out_buffer_ind, gpu_ind, deriv_above_ind;
	
	// (w_interp_ind, w_interp_shape, out_buffer_ind, gpu_ind
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &w_interp_ind, &PyTuple_Type, &w_interp_shape, 
		&deriv_above_ind, &PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &gpu_ind)) 
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
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0));
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,1));
	long C = PyLong_AsLong(PyTuple_GetItem(w_interp_shape,1));
	long M = PyLong_AsLong(PyTuple_GetItem(w_interp_shape,2));
	
	if(n_imgs*C*M*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][w_interp_ind]){
		printf("specified input sizes do not equal to stored gpu buffer %s\n", __FILE__);
		return NULL;
	}
	
	unsigned intended_sz = n_imgs*dim_above*C*N_SHIFTS*sizeof(DATA_TYPE);
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, intended_sz); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = intended_sz;
	}else if(intended_sz != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	for(int img = 0; img < n_imgs; img++){
		shift_w_dshift_out_kernel <<< dim_above, C * N_SHIFTS  >>> (gpu_buffers[gpu_ind][w_interp_ind] + img*C*M,  
			gpu_buffers[gpu_ind][deriv_above_ind] + img*dim_above*C*M,
			gpu_buffers[gpu_ind][out_buffer_ind] + img*dim_above*C*N_SHIFTS, C, M);
	}
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
