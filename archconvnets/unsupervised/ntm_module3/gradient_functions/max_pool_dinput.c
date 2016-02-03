static PyObject *max_pool_dinput(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject * imgs_shape, * deriv_above_shape;
	int imgs_ind, gpu_ind, out_buffer_ind, max_out_ind, deriv_above_ind;
	
	// src = max output
	// dest = conv output
	
	if (!PyArg_ParseTuple(args, "iO!iiO!ii", &imgs_ind, &PyTuple_Type, &imgs_shape, &max_out_ind, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
	
	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	if(out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || imgs_ind >= N_BUFFERS || imgs_ind < 0 ||
			max_out_ind >= N_BUFFERS || max_out_ind < 0 || deriv_above_ind >= N_BUFFERS || deriv_above_ind < 0){ 
		printf("buffer index incorrect\n");
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(imgs_shape,0));
	long n_channels = PyLong_AsLong(PyTuple_GetItem(imgs_shape,1));
	long img_sz = PyLong_AsLong(PyTuple_GetItem(imgs_shape,2));
	
	long n_output = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0)) / n_imgs; // repeats of deriv_above for which we will not have images (tile)
	
	if(PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0)) % n_imgs != 0){
		printf("deriv_above or imgs not correct size\n");
		return NULL;
	}
	
	int out_sz = img_sz / POOL_STRIDE;
	
	cudnnStatus_t status;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc[gpu_ind][max_out_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, out_sz, out_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(srcDiffDesc[gpu_ind][deriv_above_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, out_sz, out_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(destDesc[gpu_ind][imgs_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(destDiffDesc[gpu_ind][out_buffer_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK

	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	
	long intended_buffer_sz = n_output*n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ;
	
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
	unsigned output_ind, deriv_above_offset, out_offset;
	
	for(output_ind = 0; output_ind < n_output; output_ind++){
		out_offset = output_ind * (intended_buffer_sz / (DATA_TYPE_SZ*n_output));
		deriv_above_offset = output_ind * (n_imgs*n_channels*out_sz*out_sz);
		
		status = cudnnPoolingBackward(handle[gpu_ind], poolingDesc, srcDesc[gpu_ind][max_out_ind], gpu_buffers[gpu_ind][max_out_ind], srcDiffDesc[gpu_ind][deriv_above_ind], 
			gpu_buffers[gpu_ind][deriv_above_ind] + deriv_above_offset, destDesc[gpu_ind][imgs_ind], gpu_buffers[gpu_ind][imgs_ind],
			destDiffDesc[gpu_ind][out_buffer_ind], gpu_buffers[gpu_ind][out_buffer_ind] + out_offset);  ERR_CHECK
	}
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
