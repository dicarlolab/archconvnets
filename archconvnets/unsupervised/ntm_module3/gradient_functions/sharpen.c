#define W_SHARPEN_SZ buffer_sz[gpu_ind][w_ind]
#define GAMMA_SZ buffer_sz[gpu_ind][gamma_ind]

__global__ void sharpen_kernel(float * w, float * gamma, float * out, int dim1, int dim2){ 
	int img = blockIdx.x / dim1;
	int i = blockIdx.x % dim1;
	int j = threadIdx.x;
	int ind = img*dim1*dim2 + i*dim2 + j;

	float pow_local = __powf(w[ind], gamma[img*dim1 + i]);
	
	extern __shared__ float shared_mem[];
	float * local_sum = (float*)&shared_mem;
	
	if(j == 0)
		local_sum[0] = 0;
	__syncthreads();
	
	atomicAdd(&local_sum[0], pow_local);
	__syncthreads();

	out[ind] = pow_local / local_sum[0];
}

/*def sharpen(w, gamma):
	wg = w ** gamma
	return wg / wg.sum(1)[:,np.newaxis]*/

static PyObject * sharpen(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject * w_shape;
	int w_ind, gamma_ind, gpu_ind, out_buffer_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iii", &w_ind, &PyTuple_Type, &w_shape, &gamma_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(w_ind >= N_BUFFERS || w_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			gamma_ind >= N_BUFFERS || gamma_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(w_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem(w_shape,1));
	long dim2 = PyLong_AsLong(PyTuple_GetItem(w_shape,2));
	
	if(n_imgs*dim1*dim2*sizeof(DATA_TYPE) != W_SHARPEN_SZ || n_imgs*dim1*sizeof(DATA_TYPE) != GAMMA_SZ){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, W_SHARPEN_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = W_SHARPEN_SZ;
	}else if(W_SHARPEN_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	sharpen_kernel <<< n_imgs*dim1, dim2, sizeof(float) >>> (gpu_buffers[gpu_ind][w_ind], gpu_buffers[gpu_ind][gamma_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], dim1, dim2);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
