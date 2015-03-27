static PyObject *conv_buffers(PyObject *self, PyObject *args)  {
	cudaError_t err;
	cudnnStatus_t status;
	int i, gpu_ind, PAD, filters_ind, imgs_ind, out_ind;
	
	if (!PyArg_ParseTuple(args, "iiiii", &filters_ind, &imgs_ind, &out_ind, &PAD, &gpu_ind)) 
		return NULL;
	
	if(filters_ind >= N_BUFFERS || filters_ind < 0 || imgs_ind >= N_BUFFERS || imgs_ind < 0 || 
		out_ind >= N_BUFFERS || out_ind < 0){
		printf("invalid buffer index\n");
		return NULL;
	}
	
	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	if(data_buffers[gpu_ind][filters_ind] == NULL || data_buffers[gpu_ind][imgs_ind] == NULL){
			printf("one or more buffers not initialized on this gpu\n");
			return NULL;
	}
	
	if(filter_flags[gpu_ind][filters_ind] == 0 || filter_flags[gpu_ind][imgs_ind] == 1){
			printf("one or more buffers was not initialized correctly, filters when should be tensor or vice versa\n");
			return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	cudnnSetStream(handle, streams[gpu_ind]);
	
	int n_imgs_out;
	int n_filters_out;
	int conv_out_sz_x;
	int conv_out_sz_y;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetConvolutionDescriptor(convDesc, desc_buffers[gpu_ind][imgs_ind], desc_filters[gpu_ind][filters_ind], PAD, PAD, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);  ERR_CHECK
	
	//---------------------------------------
	// Query output layout
	//---------------------------------------
	status = cudnnGetOutputTensor4dDim(convDesc, CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK

	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	if(data_buffers[gpu_ind][out_ind] == NULL){ // allocate output
		status = cudnnCreateTensor4dDescriptor(&desc_buffers[gpu_ind][out_ind]);  ERR_CHECK
        status = cudnnSetTensor4dDescriptor(desc_buffers[gpu_ind][out_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_out, n_filters_out, conv_out_sz_x, conv_out_sz_x);  ERR_CHECK
	
		err = cudaMalloc((void**) &data_buffers[gpu_ind][out_ind], n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		data_dims[0][gpu_ind][out_ind] = n_imgs_out;
		data_dims[1][gpu_ind][out_ind] = n_filters_out;
		data_dims[2][gpu_ind][out_ind] = conv_out_sz_x;
		data_dims[3][gpu_ind][out_ind] = conv_out_sz_x;
		
		filter_flags[gpu_ind][out_ind] = 0;
	}else if(filter_flags[gpu_ind][out_ind] == 1 || data_dims[0][gpu_ind][out_ind] != n_imgs_out || 
		data_dims[1][gpu_ind][out_ind] != n_filters_out || data_dims[2][gpu_ind][out_ind] != conv_out_sz_x || 
		data_dims[3][gpu_ind][out_ind] != conv_out_sz_x){ // make sure output buffer is of correct size
			printf("output buffer size is not matching output of this function and/or initialized as a tensor, %s %i\n", __FILE__, __LINE__);
			return NULL;
	}
	
	
	//--------------------------------------
	// Convolution
	//--------------------------------------
	status = cudnnConvolutionForward(handle, desc_buffers[gpu_ind][imgs_ind], data_buffers[gpu_ind][imgs_ind], desc_filters[gpu_ind][filters_ind], data_buffers[gpu_ind][filters_ind], convDesc, desc_buffers[gpu_ind][out_ind], data_buffers[gpu_ind][out_ind], CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK
	
	cudnnSetStream(handle, NULL);
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
