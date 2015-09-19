cudnnPoolingDescriptor_t poolingDesc;

static PyObject *max_pool_cudnn(PyObject *self, PyObject *args)  {
	cudaError_t err;
	PyArrayObject *imgs_in, *vecout;
	float *imgs, *cout;
	int dims[6], gpu_ind;
	int n_channels, img_sz, n_imgs;
	
	if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &imgs_in, &gpu_ind)) 
		return NULL;
	
	if (NULL == imgs_in)  return NULL;
	
	if(gpu_ind < 0 ){//|| gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	imgs = (float *) imgs_in -> data;
	
	n_imgs = PyArray_DIM(imgs_in, 0);
	n_channels = PyArray_DIM(imgs_in, 1);
	img_sz = PyArray_DIM(imgs_in, 2);

	int out_sz = img_sz / POOL_STRIDE;
	
	float *srcData;
	float *destData;
	
	cudnnStatus_t status;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK

	//---------------------------------------
	// Query output layout
	//---------------------------------------
	//status = cudnnGetOutputTensor4dDim(convDesc, CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK

	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	status = cudnnSetTensor4dDescriptor(destDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, out_sz, out_sz); ERR_CHECK
	dims[0] = n_imgs;
	dims[1] = n_channels;
	dims[2] = out_sz;
	dims[3] = out_sz;
	
	err = cudaMalloc((void**) &destData, dims[0]*dims[1]*dims[2]*dims[3] * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	cout = (float *) vecout -> data;

	//--------------------------------------
	// allocate filter, image, alpha, and beta tensors
	//----------------------------------------
	err = cudaMalloc((void**) &srcData, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK

	//--------------------------------------
	// set filter and image values
	//--------------------------------------
	err = cudaMemcpy(srcData, imgs, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK

	//--------------------------------------
	// Pooling
	//--------------------------------------
	status = cudnnPoolingForward(handle, poolingDesc, srcDesc, srcData, destDesc, destData);  ERR_CHECK
	
	//--------------------------------------
	// Get output data
	//------------------------------------------
	err = (cudaError_t)cudaMemcpy(cout, destData, n_imgs*n_channels*out_sz*out_sz * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK

	cudaFree(destData);
	cudaFree(srcData);
	
	return PyArray_Return(vecout);
}
