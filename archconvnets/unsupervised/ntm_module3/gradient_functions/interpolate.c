__global__ void interpolate_kernel(float * interp_gate, float * o_content, float * o_prev, float * out_buffer, int dim1, int dim2, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, interp_ind, i, img;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		// we are computing the output out_buffer[img, i,j]... determine start indices of data1 & data2 for summation:
		
		img = ind_g / (dim1*dim2);
		//r = ind_g % (dim1*dim2);
		//i = r / dim2;
		i = (ind_g % (dim1*dim2)) / dim2;
		
		interp_ind = img*dim1 + i;
		
		out_buffer[ind_g] = interp_gate[interp_ind] * o_content[ind_g] + (1 - interp_gate[interp_ind]) * o_prev[ind_g];
	}
}

/*_ntm_module3.interpolate(INTERP_GATE_OUT[0], O_CONTENT[0], O_PREV[0], O_CONTENT[1], OUT_BUFFER[0], gpu_ind)
	else:
		####### CPU
		interp_gate_out = return_buffer(INTERP_GATE_OUT,gpu_ind)
		o_content = return_buffer(O_CONTENT,gpu_ind)
		o_prev = return_buffer(O_PREV,gpu_ind)
		
		OUT_BUFFER = set_buffer(interp_gate_out * o_content + (1 - interp_gate_out) * o_prev, OUT_BUFFER, gpu_ind)*/

static PyObject *interpolate(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, interp_gate_ind, o_content_ind, o_prev_ind, out_buffer_ind;
	PyObject *o_content_shape;
	
	if (!PyArg_ParseTuple(args, "iiiO!ii", &interp_gate_ind, &o_content_ind, &o_prev_ind, &PyTuple_Type, &o_content_shape,
			&out_buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(interp_gate_ind >= N_BUFFERS || interp_gate_ind < 0 || o_content_ind >= N_BUFFERS || o_content_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || o_prev_ind >= N_BUFFERS || o_prev_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(o_content_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem(o_content_shape,1));
	long dim2 = PyLong_AsLong(PyTuple_GetItem(o_content_shape,2));
	
	if(n_imgs*dim1*dim2*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][o_content_ind] || n_imgs*dim1*dim2*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][o_prev_ind] || 
			n_imgs*dim1*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][interp_gate_ind]){
		printf("specified input sizes do not equal to stored gpu buffer. dot()\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][o_content_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][o_content_ind];
	}else if(buffer_sz[gpu_ind][o_content_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}

	// determine number of blocks
	int n_blocks = (int)ceil((double)(n_imgs*dim1*dim2)/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	interpolate_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][interp_gate_ind], gpu_buffers[gpu_ind][o_content_ind],
		gpu_buffers[gpu_ind][o_prev_ind], gpu_buffers[gpu_ind][out_buffer_ind], dim1, dim2, n_imgs*dim1*dim2);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
