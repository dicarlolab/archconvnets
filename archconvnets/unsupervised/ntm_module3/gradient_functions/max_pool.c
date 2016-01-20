static PyObject *max_pool(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject * imgs_shape;
	int imgs_ind, gpu_ind, out_buffer_ind;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &imgs_ind, &PyTuple_Type, &imgs_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
	
	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	if(out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || imgs_ind >= N_BUFFERS || imgs_ind < 0){ 
		printf("buffer index incorrect\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(imgs_shape,0));
	long n_channels = PyLong_AsLong(PyTuple_GetItem(imgs_shape,1));
	long img_sz = PyLong_AsLong(PyTuple_GetItem(imgs_shape,2));

	int out_sz = img_sz / POOL_STRIDE;
	
	cudnnStatus_t status;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc[gpu_ind][imgs_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK

	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	status = cudnnSetTensor4dDescriptor(destDesc[gpu_ind][out_buffer_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, out_sz, out_sz); ERR_CHECK
	
	long intended_buffer_sz = n_imgs*n_channels*out_sz*out_sz * DATA_TYPE_SZ;
	
	if(OUT_BUFFER_SZ == 0){
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, intended_buffer_sz); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = intended_buffer_sz;
	}else if(intended_buffer_sz != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	//--------------------------------------
	// Pooling
	//--------------------------------------
	status = cudnnPoolingForward(handle[gpu_ind], poolingDesc, srcDesc[gpu_ind][imgs_ind], gpu_buffers[gpu_ind][imgs_ind], destDesc[gpu_ind][out_buffer_ind], GPU_BUFFER_OUT);  ERR_CHECK
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	PyObject *tuple = PyTuple_New(4);
	if(NULL == tuple) return NULL;
	if(-1 == PyTuple_SetItem(tuple, 0, Py_BuildValue("i", n_imgs))) return NULL;
	if(-1 == PyTuple_SetItem(tuple, 1, Py_BuildValue("i", n_channels))) return NULL;
	if(-1 == PyTuple_SetItem(tuple, 2, Py_BuildValue("i", out_sz))) return NULL;
	if(-1 == PyTuple_SetItem(tuple, 3, Py_BuildValue("i", out_sz))) return NULL;
	
	return tuple;
}
