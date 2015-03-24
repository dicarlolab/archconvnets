cudnnTensor4dDescriptor_t gradDesc_data;
cudnnFilterDescriptor_t gradDesc_filter;

static PyObject *conv_dfilter(PyObject *self, PyObject *args)  {
	cudaError_t err;
	PyArrayObject *filters_in, *imgs_in, *conv_out_in, *grad_data_in, *grad_filter_in;
	float *filters, *imgs, *conv_out;
	int i, dims[6], PAD, gpu_ind;
	int n_channels, filter_sz, n_filters, img_sz, n_imgs;
	
	if (!PyArg_ParseTuple(args, "O!O!O!ii", &PyArray_Type, &filters_in, &PyArray_Type, &imgs_in, &PyArray_Type, &conv_out_in, &PAD, &gpu_ind)) 
		return NULL;
	
	if (NULL == filters_in || NULL == imgs_in || NULL == conv_out_in)  return NULL;

	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
		
	filters = (float *) filters_in -> data;
	imgs = (float *) imgs_in -> data;
	conv_out = (float *) conv_out_in -> data;
	
	n_imgs = PyArray_DIM(imgs_in, 0);
	n_channels = PyArray_DIM(imgs_in, 1);
	img_sz = PyArray_DIM(imgs_in, 2);
	
	n_filters = PyArray_DIM(filters_in, 0);
	filter_sz = PyArray_DIM(filters_in, 2);
	
	int n_imgs_out;
	int n_filters_out;
	int conv_out_sz_x;
	int conv_out_sz_y;

	float *srcData;
	float *filterData;
	float *destData;
	float *gradData_data;
	float *gradData_filter;
	
	float *grad_data;
	float *grad_filter;
	
	cudnnStatus_t status;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(gradDesc_data, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetFilterDescriptor(filterDesc, dataType, n_filters, n_channels, filter_sz, filter_sz);  ERR_CHECK
	status = cudnnSetFilterDescriptor(gradDesc_filter, dataType, n_filters, n_channels, filter_sz, filter_sz);  ERR_CHECK
	status = cudnnSetConvolutionDescriptor(convDesc, srcDesc, filterDesc, PAD, PAD, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);  ERR_CHECK

	//---------------------------------------
	// Query output layout
	//---------------------------------------
	status = cudnnGetOutputTensor4dDim(convDesc, CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK

	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	status = cudnnSetTensor4dDescriptor(destDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs_out, n_filters_out, conv_out_sz_x, conv_out_sz_x); ERR_CHECK
	
	err = cudaMalloc((void**) &destData, n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	
	dims[0] = n_filters;
	dims[1] = n_channels;
	dims[2] = filter_sz;
	dims[3] = filter_sz;
	
	grad_filter_in=(PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	grad_filter = (float *) grad_filter_in -> data;
	
	//--------------------------------------
	// allocate filter, image, alpha, and beta tensors
	//----------------------------------------
	err = cudaMalloc((void**) &srcData, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &gradData_filter, n_filters*n_channels*filter_sz*filter_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK

	//--------------------------------------
	// set filter and image values
	//--------------------------------------
	if(n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x*DATA_TYPE_SZ != PyArray_NBYTES(conv_out_in)){
		printf("err %s %i\n", __FILE__, __LINE__);
		/*printf("%i %i\n", n_imgs, n_filters);
		printf("%i %i %i\n", n_imgs_out, n_filters_out, conv_out_sz_x);
		printf("memory error %i buffer size not expected, %i %i\n", __LINE__, n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x*DATA_TYPE_SZ, PyArray_BYTES(conv_out_in));*/
		return NULL;
	}
	err = cudaMemcpy(srcData, imgs, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(destData, conv_out, n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK

	//--------------------------------------
	// Convolution
	//--------------------------------------
	status = cudnnConvolutionBackwardFilter(handle, srcDesc, srcData, destDesc, destData, convDesc, gradDesc_filter, gradData_filter, CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK

	//--------------------------------------
	// Get output data
	//------------------------------------------
	err = (cudaError_t)cudaMemcpy(grad_filter, gradData_filter, n_filters*n_channels*filter_sz*filter_sz * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK

	cudaFree(destData);
	cudaFree(srcData);
	cudaFree(gradData_filter);
	
	return PyArray_Return(grad_filter_in);
}
