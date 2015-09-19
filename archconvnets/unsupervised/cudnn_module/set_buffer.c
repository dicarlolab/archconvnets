static PyObject *set_buffer(PyObject *self, PyObject *args){
    cudaError_t err;
	cudnnStatus_t status;
	PyArrayObject *data_in = NULL;
	float *data;
	int gpu_ind, buffer_ind, dims[5], filter_flag;
	
	if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &data_in, &buffer_ind, &filter_flag, &gpu_ind)) 
		return NULL;
        
	if (NULL == data_in)  return NULL;
    
	if(buffer_ind >= N_BUFFERS || buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
    cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	cudnnSetStream(handle, streams[gpu_ind]);
	
	dims[0] = PyArray_DIM(data_in, 0);
	dims[1] = PyArray_DIM(data_in, 1);
	dims[2] = PyArray_DIM(data_in, 2);
	dims[3] = PyArray_DIM(data_in, 3);
	
	data = (float *) data_in -> data;
	
	if(data_buffers[gpu_ind][buffer_ind] == NULL){
		//---------------------------------------
		// Set decriptor
		//---------------------------------------
		if(filter_flag){
			status = cudnnCreateFilterDescriptor(&desc_filters[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnSetFilterDescriptor(desc_filters[gpu_ind][buffer_ind], dataType, dims[0], dims[1], dims[2], dims[3]);  ERR_CHECK
		}else{
			status = cudnnCreateTensor4dDescriptor(&desc_buffers[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnSetTensor4dDescriptor(desc_buffers[gpu_ind][buffer_ind], CUDNN_TENSOR_NCHW, dataType, dims[0], dims[1], dims[2], dims[3]);  ERR_CHECK
		}
		
		//--------------------------------------
		// allocate filter, image, alpha, and beta tensors
		//----------------------------------------
		err = cudaMalloc((void**) &data_buffers[gpu_ind][buffer_ind], dims[0]*dims[1]*dims[2]*dims[3] * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		//----------------------------------------
		// save input dimensions for error checking on subsequent calls to conv()
		//---------------------------------------
		data_dims[0][gpu_ind][buffer_ind] = dims[0];
		data_dims[1][gpu_ind][buffer_ind] = dims[1];
		data_dims[2][gpu_ind][buffer_ind] = dims[2];
		data_dims[3][gpu_ind][buffer_ind] = dims[3];
		
		filter_flags[gpu_ind][buffer_ind] = filter_flag;
	}else{
		//-------------------------------------------
		// check to make sure inputs match the previously initialized buffer sizes
		//---------------------------------------------
		if(dims[0] != data_dims[0][gpu_ind][buffer_ind] || dims[1] != data_dims[1][gpu_ind][buffer_ind] || 
			dims[1] != data_dims[1][gpu_ind][buffer_ind] || dims[2] != data_dims[2][gpu_ind][buffer_ind]){
				printf("---------------------------\ninput dimensions do not match the initial input dimensions on the first call to this function\n------------------\n");
				return NULL;
		}
	}
	
	//--------------------------------------
	// set image values
	//--------------------------------------
	err = cudaMemcpy(data_buffers[gpu_ind][buffer_ind], data, dims[0]*dims[1]*dims[2]*dims[3] * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	cudnnSetStream(handle, NULL);
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
