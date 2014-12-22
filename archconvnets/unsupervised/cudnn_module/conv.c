cudnnTensor4dDescriptor_t srcDesc;
cudnnFilterDescriptor_t filterDesc;
cudnnConvolutionDescriptor_t convDesc;
cudnnTensor4dDescriptor_t destDesc;

//-------------------------------------
// conv(): perform convolution of inputs

// inputs: np raveled arrays: filters [n_filters, n_channels, filter_sz, filter_sz], imgs [n_imgs, n_channels, img_sz, img_sz]
//				ints: n_channels, filter_sz, n_filters, img_sz, n_imgs
// returns: conv_out [n_imgs, n_filters, conv_out_sz, conv_out_sz]
static PyObject *conv(PyObject *self, PyObject *args)  {
	PyArrayObject *filters_in, *imgs_in, *vecout;
	float *filters, *imgs, *cout;
	int i, dims[1];
	int n_channels, filter_sz, n_filters, img_sz, n_imgs;
	
	if (!PyArg_ParseTuple(args, "O!O!iiiii", &PyArray_Type, &filters_in, &PyArray_Type, &imgs_in, &n_channels, &filter_sz, &n_filters, &img_sz, &n_imgs)) 
		return NULL;
	if (NULL == filters || NULL == imgs)  return NULL;
	
	filters = (float *) filters_in -> data;
	imgs = (float *) imgs_in -> data;
	
	int n_imgs_out;
	int n_filters_out;
	int conv_out_sz_x;
	int conv_out_sz_y;
	cudaError_t err;

	float *srcData;
	float *filterData;
	float *destData;
	
	cudnnStatus_t status;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetFilterDescriptor(filterDesc, dataType, n_filters, n_channels, filter_sz, filter_sz);  ERR_CHECK
	status = cudnnSetConvolutionDescriptor(convDesc, srcDesc, filterDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION);  ERR_CHECK

	//---------------------------------------
	// Query output layout
	//---------------------------------------
	status = cudnnGetOutputTensor4dDim(convDesc, CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK

	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	status = cudnnSetTensor4dDescriptor(destDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs_out, n_filters_out, conv_out_sz_x, conv_out_sz_x); ERR_CHECK
	dims[0] = n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x;
	
	err = cudaMalloc((void**) &destData, dims[0] * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	cout = (float *) vecout -> data;

	//--------------------------------------
	// allocate filter, image, alpha, and beta tensors
	//----------------------------------------
	err = cudaMalloc((void**) &srcData, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &filterData, n_filters*n_channels*filter_sz*filter_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK

	//--------------------------------------
	// set filter and image values
	//--------------------------------------
	err = cudaMemcpy(srcData, imgs, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(filterData, filters, n_filters*n_channels*filter_sz*filter_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK

	//--------------------------------------
	// Convolution
	//--------------------------------------
	status = cudnnConvolutionForward(handle, srcDesc, srcData, filterDesc, filterData, convDesc, destDesc, destData, CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK

	//--------------------------------------
	// Get output data
	//------------------------------------------
	err = (cudaError_t)cudaMemcpy(cout, destData, n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK

	cudaFree(destData);
	cudaFree(srcData);
	cudaFree(filterData);
	
	return PyArray_Return(vecout);
}
