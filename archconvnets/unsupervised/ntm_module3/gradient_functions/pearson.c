__global__ void pearson_kernel(float * out, float * w1, float * w2, float * w_mean, float * BCD, int data_out_numel){
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
	
	if(ind == 0){
		out[0] = BCD[0]/sqrt(BCD[1]*BCD[2]);
	}
}

/*W1_no_mean = W1 - np.mean(W1)
W2_no_mean = W2 - np.mean(W2)

B = (W1_no_mean * W2_no_mean).sum()

C = (W1_no_mean**2).sum()
D = (W2_no_mean**2).sum()

corr = B/np.sqrt(C*D)
g2 = (W1_no_mean - (B/D)*W2_no_mean)/np.sqrt(C*D)*/

static PyObject * pearson(PyObject *self, PyObject *args){
	cudaError_t err;
	int w1_ind, w2_ind, gpu_ind, out_buffer_ind, n_imgs;
	
	if (!PyArg_ParseTuple(args, "iiiii", &w1_ind, &w2_ind, &out_buffer_ind, &n_imgs, &gpu_ind)) 
		return NULL;
    
	if(w1_ind >= N_BUFFERS || w1_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			w2_ind >= N_BUFFERS || w2_ind < 0 ){ 
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
	
	int vector_len = buffer_sz[gpu_ind][w1_ind] / (n_imgs * sizeof(DATA_TYPE));
	if(buffer_sz[gpu_ind][w1_ind] % n_imgs != 0){
		printf("n_imgs does not equally divide vector %s\n", __FILE__);
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, n_imgs*sizeof(DATA_TYPE)); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = n_imgs*sizeof(DATA_TYPE);
	}else if(n_imgs*sizeof(DATA_TYPE) != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	float * w_mean, * BCD;
	err = cudaMalloc((void**)&w_mean, 2*sizeof(DATA_TYPE)); MALLOC_ERR_CHECK
	err = cudaMalloc((void**)&BCD, 3*sizeof(DATA_TYPE)); MALLOC_ERR_CHECK
	
	for(int batch = 0; batch < n_imgs; batch++){
		// run kernel
		pearson_kernel <<< 1, MAX_THREADS_PER_BLOCK >>> (GPU_BUFFER_OUT + batch, gpu_buffers[gpu_ind][w1_ind] + batch*vector_len,
			gpu_buffers[gpu_ind][w2_ind] + batch*vector_len, w_mean, BCD, vector_len);
	}
	
	cudaFree((void**) w_mean); CHECK_CUDA_ERR
	cudaFree((void**) BCD); CHECK_CUDA_ERR
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
