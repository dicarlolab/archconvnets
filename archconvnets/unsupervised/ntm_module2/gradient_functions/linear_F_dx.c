#define DLDX(A, B, C, D) dldx[(A)*x_dim1*x_dim0*x_dim1 + (B)*x_dim0*x_dim1 + (C)*x_dim1 + D]
#define FM(A, B) F[(A)*x_dim0 + (B)]
#define DLDX_SZ (F_dim0*x_dim1*x_dim0*x_dim1*sizeof(DATA_TYPE))
#define F_SZ buffer_sz[gpu_ind][F_ind]

__global__ void linear_F_dx_kernel(float * F, float * dldx, int x_dim0, int x_dim1){ 
	int i = blockIdx.x;
	int j = threadIdx.x;

	for(int k = 0; k < x_dim1; k++){
		for(int k_local = 0; k_local < x_dim1; k_local++){
			if(k_local == k)
				DLDX(i,k,j,k_local) = FM(i,j);
			else
				DLDX(i,k,j,k_local) = 0;
		}
	}
	
	return;
}

static PyObject * linear_F_dx(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *x_shape, *F_shape;
	int F_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!O!ii", &F_ind, &PyTuple_Type, &x_shape, &PyTuple_Type, &F_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(F_ind >= N_BUFFERS || F_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(F_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long F_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)F_shape,0));
	long F_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)F_shape,1));
	long x_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,0));
	long x_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)x_shape,1));
	
	if(F_dim0*F_dim1*sizeof(DATA_TYPE) != F_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DLDX_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DLDX_SZ;
	}else if(DLDX_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	linear_F_dx_kernel <<< F_dim0, x_dim0 >>> (gpu_buffers[gpu_ind][F_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], x_dim0, x_dim1);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
