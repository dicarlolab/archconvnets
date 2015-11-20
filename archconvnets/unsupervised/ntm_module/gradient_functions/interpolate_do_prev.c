#define DIDO(A, B, C) dido[(A)*p_dim1*p_dim0*p_dim1 + (B)*p_dim0*p_dim1 + (C)*p_dim1 + D]
#define DIDO_SZ (p_dim0*p_dim1*p_dim0*p_dim1*sizeof(DATA_TYPE))
#define O_GATE_SZ buffer_sz[gpu_ind][o_gate_ind]

__global__ void interpolate_do_prev_kernel(float * keys, float * dido, int n_controllers, int mem_length){ 
	int i = threadIdx.x / n_controllers;
	int j = threadIdx.x % n_controllers;

	DIDO(i,j,i) = KEYS(i,j);
	
	for(int i_local = 0; i_local < n_controllers; i_local++){
		if(i_local != i)
			DIDO(i,j,i_local) = 0;
	}

	return;
}

static PyObject * interpolate_do_prev(PyObject *self, PyObject *args){
	cudaError_t err;
	PyTupleObject *o_gate_shape, *o_prev_shape;
	int o_gate_ind, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!O!ii", &o_gate_ind, &PyTuple_Type, &o_gate_shape, &PyTuple_Type, &o_prev_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(o_gate_ind >= N_BUFFERS || o_gate_ind < 0 ||  out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(O_GATE_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long g_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)o_gate_shape,0));
	long g_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)o_gate_shape,1));
	long p_dim0 = PyLong_AsLong(PyTuple_GetItem((PyObject *)o_prev_shape,0));
	long p_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)o_prev_shape,1));
	
	if(g_dim0*g_dim1*sizeof(DATA_TYPE) != O_GATE_SZ){
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
	
	interpolate_do_prev_kernel <<< 1, p_dim0*p_dim1 >>> (gpu_buffers[gpu_ind][o_gate_ind], 
		gpu_buffers[gpu_ind][out_buffer_ind], n_controllers, mem_length);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
