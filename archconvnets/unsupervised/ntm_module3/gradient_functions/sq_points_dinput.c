#define INPUT(A, B) input[(A)*dim1 + B]
#define OUT(A, B, C, D) out[(A)*dim1*dim0*dim1 + (B)*dim0*dim1 + (C)*dim1 + D]
#define SQ_OUT_SZ (dim0*dim1*dim0*dim1*sizeof(DATA_TYPE))

__global__ void sq_points_dinput_kernel(float * input, float * deriv_above, float * out, int dim_above, int layer_sz, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = data_out_numel / THREAD_CAPACITY;
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, img, loc;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		// we are computing the output data_out[img, dim, loc]... determine start indices of data1 & data2 for summation:
		img = ind_g / (dim_above*layer_sz);
		loc = (ind_g % (dim_above*layer_sz)) % layer_sz;
		
		out[ind_g] = 2 * input[img*layer_sz + loc] * deriv_above[ind_g];
	}
}

static PyObject *sq_points_dinput(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *input_shape;
	int input_ind, gpu_ind, out_buffer_ind, deriv_above_ind;

	if (!PyArg_ParseTuple(args, "iO!iii", &input_ind, &PyTuple_Type, &input_shape, &deriv_above_ind, &out_buffer_ind, &gpu_ind)) 
		return NULL;
		
	if(input_ind >= N_BUFFERS || input_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}

	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(input_shape,0));
	long dim_above = buffer_sz[gpu_ind][deriv_above_ind] / buffer_sz[gpu_ind][input_ind];
	long layer_sz = buffer_sz[gpu_ind][input_ind] / (n_imgs * sizeof(DATA_TYPE));
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][deriv_above_ind]); MALLOC_ERR_CHECK

		OUT_BUFFER_SZ = buffer_sz[gpu_ind][deriv_above_ind];
	}else if(OUT_BUFFER_SZ != buffer_sz[gpu_ind][deriv_above_ind]){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)buffer_sz[gpu_ind][deriv_above_ind]/(sizeof(DATA_TYPE)*MAX_THREADS_PER_BLOCK));
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	sq_points_dinput_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][input_ind], gpu_buffers[gpu_ind][deriv_above_ind], GPU_BUFFER_OUT, 
		dim_above, layer_sz, buffer_sz[gpu_ind][deriv_above_ind]/sizeof(DATA_TYPE));
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
