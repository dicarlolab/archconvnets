__global__ void relu_dlayer_in_kernel(float * layer_in, float * out, int thresh, int data_out_numel, int layer_sz, 
			int n_imgs, int dim_above){ 
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, img, loc, dim_offset, ind_temp;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		if(layer_in[ind_g] < thresh){
			img = ind_g / layer_sz;
			loc = ind_g % layer_sz;
			
			ind_temp = img*dim_above*layer_sz + loc;
			
			for(dim_offset = 0; dim_offset < dim_above*layer_sz; dim_offset += layer_sz){
				out[ind_temp + dim_offset] = 0;
			}
		}
	}
}

//_ntm_module3.relu_dlayer_in(LAYER_IN[0], DERIV_ABOVE[0], OUT_BUFFER[0], 0, gpu_ind)

static PyObject * relu_dlayer_in(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *deriv_above_shape;
	int layer_in_ind, deriv_above_ind, out_buffer_ind, gpu_ind, thresh;
	
	if (!PyArg_ParseTuple(args, "iiO!iii", &layer_in_ind, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &thresh, &gpu_ind)) 
		return NULL;
    
	if(layer_in_ind >= N_BUFFERS || layer_in_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][deriv_above_ind] <= 0 || buffer_sz[gpu_ind][layer_in_ind] <= 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape, 0));
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape, 1));
	long layer_sz = buffer_sz[gpu_ind][layer_in_ind] / (sizeof(DATA_TYPE) * n_imgs);
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][deriv_above_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][deriv_above_ind];
	}else if(buffer_sz[gpu_ind][deriv_above_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)SIGMOID_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	cudaMemcpy(gpu_buffers[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][deriv_above_ind], buffer_sz[gpu_ind][deriv_above_ind], cudaMemcpyDeviceToDevice); CHECK_CUDA_ERR
	
	relu_dlayer_in_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][layer_in_ind], gpu_buffers[gpu_ind][out_buffer_ind], 
		thresh, buffer_sz[gpu_ind][layer_in_ind]/sizeof(DATA_TYPE), layer_sz, n_imgs, dim_above);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
