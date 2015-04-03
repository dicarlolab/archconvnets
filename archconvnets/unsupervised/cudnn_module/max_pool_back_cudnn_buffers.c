cudnnTensor4dDescriptor_t srcDiffDesc;
cudnnTensor4dDescriptor_t destDiffDesc;

static PyObject *max_pool_back_cudnn_buffers(PyObject *self, PyObject *args)  {
	cudaError_t err;
	cudnnStatus_t status;
	int dims[6], gpu_ind, src_ind, src_diff_ind, dest_ind, out_ind;
	
	if (!PyArg_ParseTuple(args, "iiiii", &src_ind, &src_diff_ind, &dest_ind, &out_ind, &gpu_ind)) 
		return NULL;
	
	if(src_ind >= N_BUFFERS || src_ind < 0 || dest_ind >= N_BUFFERS || dest_ind < 0 || 
		out_ind >= N_BUFFERS || out_ind < 0 || src_diff_ind >= N_BUFFERS || src_diff_ind < 0){
		printf("invalid buffer index\n");
		return NULL;
	}
	
	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	if(data_buffers[gpu_ind][src_ind] == NULL || data_buffers[gpu_ind][dest_ind] == NULL || 
		data_buffers[gpu_ind][src_diff_ind] == NULL){
			printf("one or more buffers not initialized on this gpu\n");
			return NULL;
	}
	
	if(filter_flags[gpu_ind][src_ind] == 1 || filter_flags[gpu_ind][dest_ind] == 1 ||
		filter_flags[gpu_ind][src_diff_ind] == 1){
			printf("one or more buffers was not initialized correctly, filters when should be tensor or vice versa\n");
			return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	cudnnSetStream(handle, streams[gpu_ind]);
	
	int n_imgs = data_dims[0][gpu_ind][src_ind];
	int n_channels = data_dims[1][gpu_ind][src_ind];
	int out_sz = data_dims[2][gpu_ind][dest_ind];
	
	if(data_buffers[gpu_ind][out_ind] == NULL){ // allocate output
		status = cudnnCreateTensor4dDescriptor(&desc_buffers[gpu_ind][out_ind]);  ERR_CHECK
		status = cudnnSetTensor4dDescriptor(desc_buffers[gpu_ind][out_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, out_sz, out_sz);  ERR_CHECK
		err = cudaMalloc((void**) &data_buffers[gpu_ind][out_ind], n_imgs*n_channels*out_sz*out_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		data_dims[0][gpu_ind][out_ind] = n_imgs;
		data_dims[1][gpu_ind][out_ind] = n_channels;
		data_dims[2][gpu_ind][out_ind] = out_sz;
		data_dims[3][gpu_ind][out_ind] = out_sz;
		
		filter_flags[gpu_ind][out_ind] = 0;
	}else if(filter_flags[gpu_ind][out_ind] == 1 || data_dims[0][gpu_ind][out_ind] != n_imgs || data_dims[1][gpu_ind][out_ind] != n_channels ||
		data_dims[2][gpu_ind][out_ind] != out_sz || data_dims[3][gpu_ind][out_ind] != out_sz){ // make sure output buffer is of correct size
			printf("output buffer size is not matching output of this function and/or initialized as filters, %s %i\n", __FILE__, __LINE__);
			return NULL;
	}
	
	//--------------------------------------
	// Pooling
	//--------------------------------------
	status = cudnnPoolingBackward(handle, poolingDesc, desc_buffers[gpu_ind][src_ind], data_buffers[gpu_ind][src_ind],
		desc_buffers[gpu_ind][src_diff_ind], data_buffers[gpu_ind][src_diff_ind], 
		desc_buffers[gpu_ind][dest_ind], data_buffers[gpu_ind][dest_ind],
		desc_buffers[gpu_ind][out_ind], data_buffers[gpu_ind][out_ind]);  ERR_CHECK
	
	cudnnSetStream(handle, NULL);
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
