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
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(imgs_shape,0));
	long n_channels = PyLong_AsLong(PyTuple_GetItem(imgs_shape,1));
	long img_sz = PyLong_AsLong(PyTuple_GetItem(imgs_shape,2));
	
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,1));
	
	if(PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0)) % n_imgs != 0){
		printf("deriv_above or imgs not correct size, %s\n", __FILE__);
		return NULL;
	}
	
	int out_sz = img_sz / POOL_STRIDE;
	
	cudnnStatus_t status;
	
	long intended_sz = n_imgs * dim_above*n_channels*img_sz*img_sz * DATA_TYPE_SZ;
	
	// dim_above does not have the images stored contigously which makes this complicated
	int n_imgs_kernel = 1;
	if(dim_above == 1)
		n_imgs_kernel = n_imgs; 

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc[gpu_ind][max_out_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_kernel, n_channels, out_sz, out_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(srcDiffDesc[gpu_ind][deriv_above_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_kernel, n_channels, out_sz, out_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(destDesc[gpu_ind][imgs_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_kernel, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetTensor4dDescriptor(destDiffDesc[gpu_ind][out_buffer_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_kernel, n_channels, img_sz, img_sz);  ERR_CHECK

	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	
	if(OUT_BUFFER_SZ == 0){
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, intended_sz); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = intended_sz;
	}else if(intended_sz != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	//--------------------------------------
	// Pooling
	//--------------------------------------
	if(dim_above == 1){ // images will be contiguous with deriv_above, so we can do all images at once
	
		status = cudnnPoolingBackward(handle[gpu_ind], poolingDesc, srcDesc[gpu_ind][max_out_ind], 
				gpu_buffers[gpu_ind][max_out_ind], 
				srcDiffDesc[gpu_ind][deriv_above_ind], 
				gpu_buffers[gpu_ind][deriv_above_ind],
				destDesc[gpu_ind][imgs_ind], 
				gpu_buffers[gpu_ind][imgs_ind],
				destDiffDesc[gpu_ind][out_buffer_ind], 
				gpu_buffers[gpu_ind][out_buffer_ind]);  ERR_CHECK
	}else{
		
		unsigned deriv_above_offset, out_offset, img_offset, max_out_offset;
		
		for(int img = 0; img < n_imgs; img++){
			for(int a = 0; a < dim_above; a++){

				out_offset = img * dim_above * n_channels*img_sz*img_sz + a * n_channels*img_sz*img_sz;
				deriv_above_offset = img * dim_above * n_channels*out_sz*out_sz + a * n_channels*out_sz*out_sz;
				img_offset = img * n_channels*img_sz*img_sz;
				max_out_offset = img * n_channels*out_sz*out_sz;
			
				status = cudnnPoolingBackward(handle[gpu_ind], poolingDesc, srcDesc[gpu_ind][max_out_ind], 
					gpu_buffers[gpu_ind][max_out_ind] + max_out_offset, 
					srcDiffDesc[gpu_ind][deriv_above_ind], 
					gpu_buffers[gpu_ind][deriv_above_ind] + deriv_above_offset, 
					destDesc[gpu_ind][imgs_ind], 
					gpu_buffers[gpu_ind][imgs_ind] + img_offset,
					destDiffDesc[gpu_ind][out_buffer_ind], 
					gpu_buffers[gpu_ind][out_buffer_ind] + out_offset);  ERR_CHECK
			}
		}
	}
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
