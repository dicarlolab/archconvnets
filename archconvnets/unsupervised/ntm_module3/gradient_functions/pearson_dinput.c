// for a given correlation value, compute derivs wrt w2
__global__ void pearson_dinput_kernel(float * out, float * w1, float * w2, float * w_mean, float * BCD, int data_out_numel){
	int ind = threadIdx.x;
	
	if(ind == 0){
		w_mean[0] = 0; w_mean[1] = 0;
		BCD[0] = 0; BCD[1] = 0; BCD[2] = 0;
	}
	__syncthreads();
	
	int remainders_added;
	int min_duplicates_per_thread = data_out_numel / MAX_THREADS_PER_BLOCK;
	int n_additional_duplicates = data_out_numel % MAX_THREADS_PER_BLOCK;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates){
		n_duplicates ++;
		remainders_added = ind;
	}else
		remainders_added = n_additional_duplicates;
	
	///////////////////// compute mean
	unsigned ind_g;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup + ind*min_duplicates_per_thread + remainders_added;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		atomicAdd(&w_mean[0], w1[ind_g]/data_out_numel);
		atomicAdd(&w_mean[1], w2[ind_g]/data_out_numel);
	}

	__syncthreads();
	
	float w1_no_mean, w2_no_mean;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup + ind*min_duplicates_per_thread + remainders_added;
		
		w1_no_mean = w1[ind_g] - w_mean[0];
		w2_no_mean = w2[ind_g] - w_mean[1];
		
		atomicAdd(&BCD[0], w1_no_mean * w2_no_mean);
		atomicAdd(&BCD[1], w1_no_mean * w1_no_mean);
		atomicAdd(&BCD[2], w2_no_mean * w2_no_mean);
	}
	__syncthreads();
	
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup + ind*min_duplicates_per_thread + remainders_added;
		
		out[ind_g] = ((w1[ind_g] - w_mean[0]) - (BCD[0]/BCD[2])*(w2[ind_g] - w_mean[1]))/sqrt(BCD[1]*BCD[2]);
	}
}

/*W1_no_mean = W1 - np.mean(W1)
W2_no_mean = W2 - np.mean(W2)

B = (W1_no_mean * W2_no_mean).sum()

C = (W1_no_mean**2).sum()
D = (W2_no_mean**2).sum()

g2 = (W1_no_mean - (B/D)*W2_no_mean)/np.sqrt(C*D)*/


// multiply deriv_above with pearson_gradient
// pearson_gradient: (n_imgs, vector_len)  deriv_above (n_batches, n_imgs); 
// output: (n_batches, n_imgs, vector_len)
// out[batch] = deriv_above[batch] * pearson_gradient

__global__ void deriv_above_pearson(float * out, float * pearson_gradient, float * deriv_above, 
		int vector_len, int dim_above, int data_out_numel){

	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;

	int min_duplicates_per_thread = data_out_numel / THREAD_CAPACITY;
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;

	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;

	unsigned ind_g, batch, img, vec_i, r;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;

		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif

		//out[img, batch, vec_i] = deriv_above[img, batch] * pearson_gradient[img, vec_i];

		img = ind_g / (dim_above * vector_len);
		r = ind_g % (dim_above * vector_len);

		batch = r / vector_len;
		vec_i = r % vector_len;
		
		out[ind_g] = deriv_above[img*dim_above + batch] * 
				pearson_gradient[img*vector_len + vec_i];
	}
}

static PyObject * pearson_dinput(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject * deriv_above_shape;
	int w1_ind, w2_ind, gpu_ind, out_buffer_ind, deriv_above_ind;
	
	if (!PyArg_ParseTuple(args, "iiiiO!i", &w1_ind, &w2_ind, &out_buffer_ind, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape, 
			&gpu_ind)) 
		return NULL;
    
	if(w1_ind >= N_BUFFERS || w1_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			w2_ind >= N_BUFFERS || w2_ind < 0 || deriv_above_ind >= N_BUFFERS || deriv_above_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(buffer_sz[gpu_ind][w1_ind] != buffer_sz[gpu_ind][w2_ind] || buffer_sz[gpu_ind][w1_ind] == 0){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0));
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,1));
	
	int vector_len = buffer_sz[gpu_ind][w1_ind] / (n_imgs * sizeof(DATA_TYPE));
	if(buffer_sz[gpu_ind][w1_ind] % n_imgs != 0){
		printf("n_imgs does not equally divide vector %s\n", __FILE__);
		return NULL;
	}
	
	unsigned intended_sz = dim_above*buffer_sz[gpu_ind][w1_ind];

	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, intended_sz); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = intended_sz;
	}else if(intended_sz != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	float * w_mean, * BCD, * pearson_grad;
	err = cudaMalloc((void**) &w_mean, 2*sizeof(DATA_TYPE)); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &BCD, 3*sizeof(DATA_TYPE)); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &pearson_grad, buffer_sz[gpu_ind][w1_ind]);

	int offset, n_blocks;
	for(int img = 0; img < n_imgs; img++){
		// run kernel
		offset = img*vector_len;
		pearson_dinput_kernel <<< 1, MAX_THREADS_PER_BLOCK >>> (pearson_grad + offset, 
			gpu_buffers[gpu_ind][w1_ind] + offset, gpu_buffers[gpu_ind][w2_ind] + offset, w_mean, BCD, vector_len);
	}
	
	// determine number of blocks
	n_blocks = (int)ceil((double)(n_imgs*vector_len*dim_above)/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;

	 deriv_above_pearson <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (GPU_BUFFER_OUT, pearson_grad, 
	 	gpu_buffers[gpu_ind][deriv_above_ind], vector_len, dim_above, n_imgs*vector_len*dim_above);
	

	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaFree(pearson_grad); CHECK_CUDA_ERR
	cudaFree(w_mean); CHECK_CUDA_ERR
	cudaFree(BCD); CHECK_CUDA_ERR
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
