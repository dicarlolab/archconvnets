#define DIDO_SZ (n_imgs*dim_above*dim0*dim1*sizeof(DATA_TYPE))

__global__ void interpolate_do_prev_kernel(float * interp_gate_out, float * deriv_above, float * out_data, int dim0, int dim1,
		int dim_above){ 
	int img = blockIdx.x / dim_above;
	int a = blockIdx.x % dim_above;
	int i = threadIdx.x / dim1;
	int j = threadIdx.x % dim1;

	unsigned ind = img*dim_above*dim0*dim1 + a*dim0*dim1 + i*dim1 + j;

	out_data[ind] = deriv_above[ind] * (1-interp_gate_out[img*dim0 + i]);
}

static PyObject * interpolate_do_prev(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *o_prev_shape;
	int interp_gate_out_ind, out_buffer_ind, deriv_above_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iii", &interp_gate_out_ind, &PyTuple_Type, &o_prev_shape,
		&deriv_above_ind, &out_buffer_ind, &gpu_ind)) 
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
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(o_prev_shape,0));
	long dim0 = PyLong_AsLong(PyTuple_GetItem(o_prev_shape,1));
	long dim1 = PyLong_AsLong(PyTuple_GetItem(o_prev_shape,2));
	
	long dim_above = buffer_sz[gpu_ind][deriv_above_ind] / (n_imgs*dim0*dim1*sizeof(DATA_TYPE));
	
	if(n_imgs*dim0*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][interp_gate_out_ind]){
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
	
	interpolate_do_prev_kernel <<< n_imgs*dim_above, dim0*dim1 >>> (gpu_buffers[gpu_ind][interp_gate_out_ind],
		gpu_buffers[gpu_ind][deriv_above_ind],	
		gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1, dim_above);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
