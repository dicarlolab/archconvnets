static PyObject * conv(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject * filters_shape, * imgs_shape;
	int filters_ind, imgs_ind, PAD, out_buffer_ind, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iO!iii", &filters_ind, &PyTuple_Type, &filters_shape, &imgs_ind, &PyTuple_Type, &imgs_shape, &PAD, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(filters_ind >= N_BUFFERS || filters_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			imgs_ind >= N_BUFFERS || imgs_ind < 0){ 
		printf("buffer index incorrect\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect\n");
		return NULL;
	}
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(imgs_shape,0));
	long n_channels = PyLong_AsLong(PyTuple_GetItem(imgs_shape,1));
	long img_sz = PyLong_AsLong(PyTuple_GetItem(imgs_shape,2));
	
	long n_filters = PyLong_AsLong(PyTuple_GetItem(filters_shape,0));
	long filter_sz = PyLong_AsLong(PyTuple_GetItem(filters_shape,2));
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	int n_imgs_out;
	int n_filters_out;
	int conv_out_sz_x;
	int conv_out_sz_y;

	cudnnStatus_t status;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc[gpu_ind][imgs_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetFilterDescriptor(filterDesc[gpu_ind][filters_ind], dataType, n_filters, n_channels, filter_sz, filter_sz);  ERR_CHECK
	status = cudnnSetConvolutionDescriptor(convDesc[gpu_ind][out_buffer_ind], srcDesc[gpu_ind][imgs_ind], filterDesc[gpu_ind][filters_ind], PAD, PAD, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);  ERR_CHECK

	//---------------------------------------
	// Query output layout
	//---------------------------------------
	status = cudnnGetOutputTensor4dDim(convDesc[gpu_ind][out_buffer_ind], CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK

	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	status = cudnnSetTensor4dDescriptor(destDesc[gpu_ind][out_buffer_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_out, n_filters_out, conv_out_sz_x, conv_out_sz_x); ERR_CHECK
	
	long intended_buffer_sz = n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x * DATA_TYPE_SZ;
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, intended_buffer_sz); MALLOC_ERR_CHECK
	
		OUT_BUFFER_SZ = intended_buffer_sz;
	}else if(intended_buffer_sz != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	//--------------------------------------
	// Convolution
	//--------------------------------------
	status = cudnnConvolutionForward(handle[gpu_ind], srcDesc[gpu_ind][imgs_ind], gpu_buffers[gpu_ind][imgs_ind], 
		filterDesc[gpu_ind][filters_ind], gpu_buffers[gpu_ind][filters_ind], convDesc[gpu_ind][out_buffer_ind], destDesc[gpu_ind][out_buffer_ind], GPU_BUFFER_OUT, CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK

	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	PyObject *tuple = PyTuple_New(4);
	if(NULL == tuple) return NULL;
	if(-1 == PyTuple_SetItem(tuple, 0, Py_BuildValue("i", n_imgs_out))) return NULL;
	if(-1 == PyTuple_SetItem(tuple, 1, Py_BuildValue("i", n_filters_out))) return NULL;
	if(-1 == PyTuple_SetItem(tuple, 2, Py_BuildValue("i", conv_out_sz_x))) return NULL;
	if(-1 == PyTuple_SetItem(tuple, 3, Py_BuildValue("i", conv_out_sz_y))) return NULL;
	
	return tuple;
}
