#define W(A, B) w[(A)*dim1 + B]
#define DSDG_SZ (dim0*dim1*dim0*sizeof(DATA_TYPE))
#define W_SZ buffer_sz[gpu_ind][w_ind]
#define GAMMA_SZ buffer_sz[gpu_ind][gamma_ind]
#define WG_SUM_GAMMA shared[0]
#define LN_W_WG_SUM shared[1]

__global__ void dsharpen_dgamma_kernel(float * w, float * gamma, float * dsdg, int dim0, int dim1){ 
	int i = blockIdx.x;
	int j = threadIdx.x;

	extern __shared__ float shared_mem[];
	float * shared = (float*)&shared_mem;
	
	float wg, ln_w_wg;

	WG_SUM_GAMMA = 0;
	LN_W_WG_SUM = 0;
	__syncthreads();
	wg = pow(W(i,j), gamma[i]);
	ln_w_wg = log(W(i,j)) * wg;

	atomicAdd(&WG_SUM_GAMMA, wg);
	atomicAdd(&LN_W_WG_SUM, ln_w_wg);
	__syncthreads();

	DSDG(i,j,i) = (ln_w_wg * WG_SUM_GAMMA - wg * LN_W_WG_SUM) / (WG_SUM_GAMMA * WG_SUM_GAMMA);

	for(int i_local = 0; i_local < dim0; i_local++){
		if(i_local != i)
			DSDG(i,j,i_local) = 0;
	}
	
	return;
}

static PyObject *dsharpen_dgamma(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *w_shape, *gamma_shape;
	int w_ind, gamma_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &w_ind, &PyTuple_Type, &w_shape,
		&gamma_ind, &PyTuple_Type, &gamma_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(w_ind >= N_BUFFERS || w_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || 
			gamma_ind >= N_BUFFERS || gamma_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(GAMMA_SZ == 0 || W_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)w_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)w_shape,1));
	long dim0_gamma = PyLong_AsLong(PyTuple_GetItem((PyObject *)gamma_shape,0));
	
	if(dim0*dim1*sizeof(DATA_TYPE) != W_SZ || dim0*sizeof(DATA_TYPE) != GAMMA_SZ){
		printf("specified input sizes do not equal to stored gpu buffer. dot_cpu()\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DSDG_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DSDG_SZ;
	}else if(DSDG_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	dsharpen_dgamma_kernel <<< dim0, dim1, sizeof(float)*2 >>> (gpu_buffers[gpu_ind][w_ind], gpu_buffers[gpu_ind][gamma_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}

