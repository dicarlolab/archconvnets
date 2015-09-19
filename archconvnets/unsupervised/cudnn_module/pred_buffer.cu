__global__ void kernel_pred_buffer(float * FL, float * max_output3, float * pred, int N_C, int n_imgs, int n_filters, 
		int conv_out_sz, uint8_t * Y){
	int cat = blockIdx.x;
	int img = blockIdx.y;
	
    int n3 = threadIdx.x;
	int z1 = threadIdx.y;
	
	int pred_ind = cat*n_imgs + img;
	
	if(n3 == 0)
		pred[pred_ind] = 0;
	__syncthreads();
	
	int z;
	
	float temp_sum = 0;
	
	int FL_ind = cat*n_filters*conv_out_sz*conv_out_sz + n3*conv_out_sz*conv_out_sz + z1*conv_out_sz;
	int max_output3_ind = img*n_filters*conv_out_sz*conv_out_sz + n3*conv_out_sz*conv_out_sz + z1*conv_out_sz;
	
	/*for(z = 0; z < conv_out_sz*conv_out_sz; z++){
		temp_sum += FL[FL_ind + z] * max_output3[max_output3_ind + z];
	}*/
	
	for(z = 0; z < conv_out_sz; z++){
		temp_sum += FL[FL_ind + z] * max_output3[max_output3_ind + z];
	}
	
	atomicAdd(&pred[pred_ind], temp_sum);
	
	__syncthreads();
	if(n3 == 0)
		pred[pred_ind] -= Y[pred_ind];
	
	return;
}

// pred === np.einsum(FL, range(4), max_output3, [4,1,2,3], [0,4]) - Ys
// it is the category predictions for each image [N_C x n_imgs] minus the labels, given FL, max_output3, and Y
static PyObject *pred_buffer(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, FL_IND, out_ind, max3_ind;
	PyArrayObject *Y_in = NULL;
	uint8_t *Y, *Y_c;
	
	if (!PyArg_ParseTuple(args, "iiiO!i", &FL_IND, &max3_ind, &out_ind, &PyArray_Type, &Y_in, &gpu_ind)) 
		return NULL;
	
	if(Y_in == NULL) return NULL;
	
	if(FL_IND >= N_BUFFERS || FL_IND < 0 || max3_ind >= N_BUFFERS || max3_ind < 0 || 
		out_ind >= N_BUFFERS || out_ind < 0){
		printf("invalid buffer index\n");
		return NULL;
	}
	
	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	if(data_buffers[gpu_ind][FL_IND] == NULL || data_buffers[gpu_ind][max3_ind] == NULL){
			printf("one or more buffers not initialized on this gpu\n");
			return NULL;
	}
	
	if(filter_flags[gpu_ind][FL_IND] == 0 || filter_flags[gpu_ind][max3_ind] == 1){
			printf("one or more buffers was not initialized correctly, filters when should be tensor or vice versa\n");
			return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	int n_imgs_out = data_dims[0][gpu_ind][max3_ind];
	int N_C = data_dims[0][gpu_ind][FL_IND];
	int n_filters_out = data_dims[1][gpu_ind][FL_IND];
	int conv_out_sz = data_dims[2][gpu_ind][max3_ind];
	
	Y = (uint8_t *) Y_in -> data;
	
	if(PyArray_DIM(Y_in, 0) != N_C || PyArray_DIM(Y_in, 1) != n_imgs_out){
		printf("shape of Y is not matching max3_ind or FL buffers\n");
		return NULL;
	}
	
	// put labels on GPU
	err = cudaMalloc((void**) &Y_c, N_C*n_imgs_out * sizeof(uint8_t)); MALLOC_ERR_CHECK
	err = cudaMemcpy(Y_c, Y, N_C*n_imgs_out * sizeof(uint8_t), cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	if(data_2d_buffers[gpu_ind][out_ind] == NULL){ // allocate output
		err = cudaMalloc((void**) &data_2d_buffers[gpu_ind][out_ind], N_C*n_imgs_out * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		data_2d_dims[0][gpu_ind][out_ind] = N_C;
		data_2d_dims[1][gpu_ind][out_ind] = n_imgs_out;
		
	}else if(data_2d_dims[0][gpu_ind][out_ind] != N_C || 
		data_2d_dims[1][gpu_ind][out_ind] != n_imgs_out){ // make sure output buffer is of correct size
			printf("output buffer size is not matching output of this function and/or initialized as a tensor, %s %i\n", __FILE__, __LINE__);
			return NULL;
	}
	
	/////////////////
	dim3 thread_sz, grid_sz;
	
	grid_sz.x = N_C;
	grid_sz.y = n_imgs_out;
	
	thread_sz.x = n_filters_out;
	thread_sz.y = conv_out_sz;
	
	kernel_pred_buffer <<< grid_sz, thread_sz >>> (data_buffers[gpu_ind][FL_IND], data_buffers[gpu_ind][max3_ind], data_2d_buffers[gpu_ind][out_ind],
		N_C, n_imgs_out, n_filters_out, conv_out_sz, Y_c);
	
	cudaFree(Y_c);
	
	Py_INCREF(Py_None);
	return Py_None;
}
