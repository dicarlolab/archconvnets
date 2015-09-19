cudnnTensor4dDescriptor_t srcDiffDesc;
cudnnTensor4dDescriptor_t destDiffDesc;

static PyObject *max_pool_back_cudnn(PyObject *self, PyObject *args)  {
	cudaError_t err;
	PyArrayObject *vecout, *srcData_in, *srcDiffData_in, *destData_in;
	float *cout, *srcData_c, *srcData, *srcDiffData_c, *srcDiffData, *destDiffData_c, *destData_c, *destData;
	int dims[6], gpu_ind;
	int n_channels, out_sz, img_sz, n_imgs;
	
	if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &srcData_in, &PyArray_Type, &srcDiffData_in, &PyArray_Type, &destData_in, &gpu_ind)) 
		return NULL;
	
	if (NULL == srcData_in || NULL == srcDiffData_in || NULL == destData_in)  return NULL;
	
	if(gpu_ind < 0 ){//|| gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	srcData = (float *) srcData_in -> data;
	srcDiffData = (float *) srcDiffData_in -> data;
	destData = (float *) destData_in -> data;
	
	n_imgs = PyArray_DIM(srcData_in, 0);
	n_channels = PyArray_DIM(srcData_in, 1);
	img_sz = PyArray_DIM(srcData_in, 2);
	
	out_sz = PyArray_DIM(destData_in, 2);

	cudnnStatus_t status;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(srcDiffDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(destDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, out_sz, out_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(destDiffDesc, CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, out_sz, out_sz);  ERR_CHECK

	
	dims[0] = n_imgs;
	dims[1] = n_channels;
	dims[2] = out_sz;
	dims[3] = out_sz;
	
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	cout = (float *) vecout -> data;

	//--------------------------------------
	// allocate filter, image, alpha, and beta tensors
	//----------------------------------------
	err = cudaMalloc((void**) &srcData_c, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &srcDiffData_c, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &destData_c, n_imgs*n_channels*out_sz*out_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &destDiffData_c, n_imgs*n_channels*out_sz*out_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK

	//--------------------------------------
	// set filter and image values
	//--------------------------------------
	err = cudaMemcpy(srcData_c, srcData, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(srcDiffData_c, srcDiffData, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(destData_c, destData, n_imgs*n_channels*out_sz*out_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK

	//--------------------------------------
	// Pooling
	//--------------------------------------
	status = cudnnPoolingBackward(handle, poolingDesc, srcDesc, srcData_c, srcDiffDesc, srcDiffData_c, destDesc, destData_c,
		destDiffDesc, destDiffData_c);  ERR_CHECK
	
	//--------------------------------------
	// Get output data
	//------------------------------------------
	err = (cudaError_t)cudaMemcpy(cout, destDiffData_c, n_imgs*n_channels*out_sz*out_sz * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK
	
	cudaFree(destDiffData_c);
	cudaFree(destData_c);
	cudaFree(srcData_c);
	cudaFree(srcDiffData_c);
	
	return PyArray_Return(vecout);
}
