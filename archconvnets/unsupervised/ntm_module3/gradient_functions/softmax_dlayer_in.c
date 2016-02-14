#define LAYER_OUT(A, B) layer_out[(A)*dim1 + B]
#define LAYER_OUT_SZ buffer_sz[gpu_ind][layer_out_ind]

__global__ void softmax_dlayer_in_kernel(float * layer_out, float * deriv_above, float * out, long dim0, long dim1){
	int a = blockIdx.x;
	int b = threadIdx.x / dim1;
	int c = threadIdx.x % dim1;
	
	int ind = a*dim0*dim1 + b*dim1 + c;
	
	out[ind] = 0;
	for(int c1 = 0; c1 < dim1; c1++){
		if(c1 == c)
			out[ind] += LAYER_OUT(b,c1) * (1 - LAYER_OUT(b,c1)) * deriv_above[a*dim0*dim1 + b*dim1 + c1];
		else
			out[ind] += -LAYER_OUT(b,c1)*LAYER_OUT(b,c) * deriv_above[a*dim0*dim1 + b*dim1 + c1];
	}
	
	//out[ind] *= deriv_above[ind];
}

// [a,b,c] * [b,c,b,c] = [a,b,c]
// [a,b,c1] * [b,c1,c] = [a,b,c]


static PyObject *softmax_dlayer_in(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *layer_out_shape;
	int layer_out_ind, gpu_ind, out_buffer_ind, deriv_above_ind;

	if (!PyArg_ParseTuple(args, "iO!iii", &layer_out_ind, &PyTuple_Type, &layer_out_shape, &deriv_above_ind,
			&out_buffer_ind, &gpu_ind)) 
		return NULL;
		
	if(layer_out_ind >= N_BUFFERS || layer_out_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}

	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(LAYER_OUT_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}

	long dim0 = PyLong_AsLong(PyTuple_GetItem(layer_out_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem(layer_out_shape,1));

	if(dim0*dim1*sizeof(DATA_TYPE) != LAYER_OUT_SZ){
		printf("specified input sizes do not equal stored gpu buffer. softmax_dlayer_in()\n");
		return NULL;
	}
	
	if((buffer_sz[gpu_ind][deriv_above_ind] % (dim0*dim1)) != 0){
		printf("deriv_above incorrect size %s\n", __FILE__);
		return NULL;
	}
	int n_batches = buffer_sz[gpu_ind][deriv_above_ind] / (dim0*dim1);
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][deriv_above_ind]); MALLOC_ERR_CHECK

		OUT_BUFFER_SZ = buffer_sz[gpu_ind][deriv_above_ind];
	}else if(buffer_sz[gpu_ind][deriv_above_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}

	softmax_dlayer_in_kernel <<< n_batches, dim0*dim1 >>> (gpu_buffers[gpu_ind][layer_out_ind], gpu_buffers[gpu_ind][deriv_above_ind],
			GPU_BUFFER_OUT, dim0, dim1);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
