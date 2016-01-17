__global__ void sum_points_kernel(float * points, float * out){
	if(threadIdx.x == 0) out[0] = 0;
	__syncthreads();
	atomicAdd(&out[0], points[threadIdx.x]);
}

static PyObject * sum_points(PyObject *self, PyObject *args){
	cudaError_t err;
	int points_ind, points_len, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iiii", &points_ind, &points_len, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(points_ind >= N_BUFFERS || points_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(points_len*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][points_ind]){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, sizeof(DATA_TYPE)); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = sizeof(DATA_TYPE);
	}else if(sizeof(DATA_TYPE) != buffer_sz[gpu_ind][out_buffer_ind]){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	sum_points_kernel <<< 1, points_len >>> (gpu_buffers[gpu_ind][points_ind], gpu_buffers[gpu_ind][out_buffer_ind]);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
