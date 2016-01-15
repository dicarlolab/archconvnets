static PyObject *conv_ddata_buffers(PyObject *self, PyObject *args)  {
	cudaError_t err;
	cudnnStatus_t status;
	int gpu_ind, PAD, filters_ind, imgs_ind, conv_out_ind, out_ind;
	
	if (!PyArg_ParseTuple(args, "iiiiii", &filters_ind, &imgs_ind, &conv_out_ind, &out_ind, &PAD, &gpu_ind)) 
		return NULL;
	
	if(filters_ind >= N_BUFFERS || filters_ind < 0 || imgs_ind >= N_BUFFERS || imgs_ind < 0 || 
		conv_out_ind >= N_BUFFERS || conv_out_ind < 0 || out_ind >= N_BUFFERS || out_ind < 0){
		printf("invalid buffer index\n");
		return NULL;
	}
	
	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	if(data_buffers[gpu_ind][filters_ind] == NULL || data_buffers[gpu_ind][imgs_ind] == NULL || 
		data_buffers[gpu_ind][conv_out_ind] == NULL){
			printf("one or more buffers not initialized on this gpu\n");
			return NULL;
	}
	
	if(filter_flags[gpu_ind][filters_ind] == 0 || filter_flags[gpu_ind][imgs_ind] == 1 ||
		filter_flags[gpu_ind][conv_out_ind] == 1){
			printf("one or more buffers was not initialized correctly, filters when should be tensor or vice versa\n");
			return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	cudnnSetStream(handle, streams[gpu_ind]);
    
	int n_imgs = data_dims[0][gpu_ind][conv_out_ind];
	int n_channels = data_dims[1][gpu_ind][imgs_ind];
	int img_sz = data_dims[2][gpu_ind][imgs_ind];
	
	if(data_buffers[gpu_ind][out_ind] == NULL){ // allocate output
		status = cudnnCreateTensor4dDescriptor(&desc_buffers[gpu_ind][out_ind]);  ERR_CHECK
        status = cudnnSetTensor4dDescriptor(desc_buffers[gpu_ind][out_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
        
		err = cudaMalloc((void**) &data_buffers[gpu_ind][out_ind], n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		data_dims[0][gpu_ind][out_ind] = n_imgs;
		data_dims[1][gpu_ind][out_ind] = n_channels;
		data_dims[2][gpu_ind][out_ind] = img_sz;
		data_dims[3][gpu_ind][out_ind] = img_sz;
		
		filter_flags[gpu_ind][out_ind] = 0;
	}else if(filter_flags[gpu_ind][out_ind] == 1 || data_dims[0][gpu_ind][out_ind] != n_imgs || 
		data_dims[1][gpu_ind][out_ind] != n_channels || data_dims[2][gpu_ind][out_ind] != img_sz || 
		data_dims[3][gpu_ind][out_ind] != img_sz){ // make sure output buffer is of correct size
			printf("output buffer size is not matching output of this function and/or initialized as a tensor, %s %i\n", __FILE__, __LINE__);
			return NULL;
	}
    
	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetConvolutionDescriptor(convDesc, desc_buffers[gpu_ind][imgs_ind], desc_filters[gpu_ind][filters_ind], PAD, PAD, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);  ERR_CHECK

	//---------------------------------------
	// Query output layout
	//---------------------------------------
    int n_imgs_out, n_filters_out, conv_out_sz_x, conv_out_sz_y;
	status = cudnnGetOutputTensor4dDim(convDesc, CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK

	/*if(n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x != data_dims[0][gpu_ind][conv_out_ind]*data_dims[1][gpu_ind][conv_out_ind]*
		data_dims[2][gpu_ind][conv_out_ind]*data_dims[3][gpu_ind][conv_out_ind]){
		printf("predicted conv output not matching given input %s %i\n", __FILE__, __LINE__);
		printf("%i %i\n", n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x, data_dims[0][gpu_ind][conv_out_ind]*data_dims[1][gpu_ind][conv_out_ind]*
		data_dims[2][gpu_ind][conv_out_ind]*data_dims[3][gpu_ind][conv_out_ind]);
		printf("%i %i\n", n_imgs_out, data_dims[0][gpu_ind][conv_out_ind]);
		printf("%i %i\n", n_filters_out, data_dims[1][gpu_ind][conv_out_ind]);
		printf("%i %i\n", conv_out_sz_x, data_dims[2][gpu_ind][conv_out_ind]);
		printf("%i %i\n", conv_out_sz_y, data_dims[3][gpu_ind][conv_out_ind]);
		return NULL;
	}*/
    
	
	//--------------------------------------
	// Convolution
	//--------------------------------------
	status = cudnnConvolutionBackwardData(handle, desc_filters[gpu_ind][filters_ind], 
    data_buffers[gpu_ind][filters_ind], desc_buffers[gpu_ind][conv_out_ind], 
    data_buffers[gpu_ind][conv_out_ind], convDesc, 
    desc_buffers[gpu_ind][out_ind], 
    data_buffers[gpu_ind][out_ind], CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK
    
    cudnnSetStream(handle, NULL);
	cudaSetDevice(0); CHECK_CUDA_ERR
    
	Py_INCREF(Py_None);
	return Py_None;
}
