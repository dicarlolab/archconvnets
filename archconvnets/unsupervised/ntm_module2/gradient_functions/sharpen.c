#define W_SHARPEN_SZ buffer_sz[gpu_ind][w_ind]
#define GAMMA_SZ buffer_sz[gpu_ind][gamma_ind]

__global__ void sharpen_kernel(float * w, float * gamma, float * out, int dim1, int dim2){ 
	int i = threadIdx.x / dim2;

	out[threadIdx.x] = __powf(w[i], gamma[i]);
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
	long dim1 = PyLong_AsLong(PyTuple_GetItem(w_shape,0));
	long dim2 = PyLong_AsLong(PyTuple_GetItem(w_shape,1));
	
	if(dim1*dim2*sizeof(DATA_TYPE) != W_SHARPEN_SZ || dim1*sizeof(DATA_TYPE) != GAMMA_SZ){
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
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	sharpen_kernel <<< 1, dim1 * dim2 >>> (gpu_buffers[gpu_ind][w_ind], gpu_buffers[gpu_ind][gamma_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], dim1, dim2);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
