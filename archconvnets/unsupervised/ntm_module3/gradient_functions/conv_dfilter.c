/*
# deriv_above [n_imgs, dim_above, n_filters, out_sz, out_sz]
# F [n_filters, 3, f_sz, f_sz]
# imgs [n_imgs, 3, in_sz, in_sz]

# dF = [n_imgs, n_filters, out_sz, out_sz, n_filters, 3, f_sz, f_sz]

# ... dF = [n_imgs, out_sz, out_sz, 3, f_sz, f_sz]

# deriv_above * dF = [n_imgs, dim_above, n_filters, 3, f_sz, f_sz]

dF = np.zeros((n_imgs, out_sz, out_sz, 3, filter_sz, filter_sz), dtype='single')

for loc_x in range(out_sz):
	for loc_y in range(out_sz):
		for f1 in range(filter_sz):
			for f2 in range(filter_sz):
				dF[:, loc_x, loc_y, :, f1, f2] = imgs_pad[:,:, loc_x + f1, loc_y + f2]

dF_cpu = np.einsum(deriv_above, range(5), dF, [0, 3,4, 5,6,7], [0,1,2, 5,6,7])
         sum across out_sz dims*/

__global__ void conv_dfilter_nsum(float * imgs, float * deriv_above, float * out, int dim_above, int n_filters,
		int out_sz, int n_channels, int f_sz, int img_sz, int PAD, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, ind_deriv, ind_img, o1, o2, img, a, f, f1, f2, r, c;
	unsigned ind_deriv_temp, ind_img_temp;
	
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		img = ind_g / (dim_above * n_filters * n_channels * f_sz * f_sz);
		r = ind_g % (dim_above * n_filters * n_channels * f_sz * f_sz);
		
		a = r / (n_filters * n_channels * f_sz * f_sz);
		r = r % (n_filters * n_channels * f_sz * f_sz);
		
		f = r / (n_channels * f_sz * f_sz);
		r = r % (n_channels * f_sz * f_sz);
		
		c = r / (f_sz * f_sz);
		r = r % (f_sz * f_sz);
		
		f1 = r / f_sz;
		f2 = r % f_sz;
		
		ind_deriv_temp = img * dim_above*n_filters*out_sz*out_sz + a * n_filters*out_sz*out_sz + f * out_sz*out_sz;
		ind_img_temp = img * n_channels*img_sz*img_sz + c * img_sz*img_sz + (f1 - PAD)*img_sz + f2 - PAD;
		
		out[ind_g] = 0;
		//for(o1 = 0; o1 < out_sz; o1++){
		for(o1 = 0; o1 < out_sz; o1++){
			if ((o1 + f1 - PAD) < img_sz)
				for(o2 = 0; o2 < out_sz; o2++){
					/*# deriv_above [n_imgs, dim_above, n_filters, out_sz, out_sz]
					# imgs [n_imgs, 3, in_sz, in_sz]*/
					if ((o1 + f1 - PAD) < img_sz && (o2 + f2 - PAD) < img_sz){
						ind_deriv = ind_deriv_temp + o1 * out_sz + o2;
						
						ind_img = ind_img_temp + o1*img_sz + o2;
						
						out[ind_g] += deriv_above[ind_deriv] * imgs[ind_img];
					}
				} // o1
		} //o2
	} // ind_g
}

static PyObject * conv_dfilter(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject * filters_shape, * imgs_shape, * deriv_above_shape;
	int filters_ind, PAD, out_buffer_ind, deriv_above_ind, gpu_ind, imgs_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iO!iO!iii", &filters_ind, &PyTuple_Type, &filters_shape, &imgs_ind, &PyTuple_Type, &imgs_shape, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape, &PAD, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(filters_ind >= N_BUFFERS || filters_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			deriv_above_ind >= N_BUFFERS || deriv_above_ind < 0){ 
		printf("buffer index incorrect\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect\n");
		return NULL;
	}
	
	// get sizes
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,1));
	
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(imgs_shape,0));
	long n_channels = PyLong_AsLong(PyTuple_GetItem(imgs_shape,1));
	long img_sz = PyLong_AsLong(PyTuple_GetItem(imgs_shape,2));
	
	long n_filters = PyLong_AsLong(PyTuple_GetItem(filters_shape,0));
	long filter_sz = PyLong_AsLong(PyTuple_GetItem(filters_shape,2));
	
	long n_batches = n_imgs * dim_above;
	
	int n_imgs_out;
	int n_filters_out;
	int conv_out_sz_x;
	int conv_out_sz_y;

	long intended_sz = n_filters*n_channels*filter_sz*filter_sz * DATA_TYPE_SZ;
	int n_imgs_kernel = n_imgs; // only if dim_above == 1
	
	if(dim_above != 1){ // don't sum across images
		intended_sz *= n_batches;
		
		n_imgs_kernel = 1; // compute 1 image at a time (since cudnn sums gradients from multiple imgs)
	}
	
	cudnnStatus_t status;

	//---------------------------------------
	// Set decriptors
	//---------------------------------------
	status = cudnnSetTensor4dDescriptor(srcDesc[gpu_ind][imgs_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_kernel, n_channels, img_sz, img_sz);  ERR_CHECK
	status = cudnnSetFilterDescriptor(filterDesc[gpu_ind][filters_ind], dataType, n_filters, n_channels, filter_sz, filter_sz);  ERR_CHECK
	status = cudnnSetFilterDescriptor(gradDesc_filter[gpu_ind][out_buffer_ind], dataType, n_filters, n_channels, filter_sz, filter_sz);  ERR_CHECK
	status = cudnnSetConvolutionDescriptor(convDesc[gpu_ind][out_buffer_ind], srcDesc[gpu_ind][imgs_ind], filterDesc[gpu_ind][filters_ind], PAD, PAD, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);  ERR_CHECK
	
	//---------------------------------------
	// Query output layout
	//---------------------------------------
	status = cudnnGetOutputTensor4dDim(convDesc[gpu_ind][out_buffer_ind], CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK
	
	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	status = cudnnSetTensor4dDescriptor(destDesc[gpu_ind][deriv_above_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_kernel, n_filters_out, conv_out_sz_x, conv_out_sz_x); ERR_CHECK
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, intended_sz); MALLOC_ERR_CHECK
	
		OUT_BUFFER_SZ = intended_sz;
	}else if(intended_sz != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	//--------------------------------------
	// Convolution
	//--------------------------------------
	if(dim_above != 1){ // don't sum imgs
		/////////////////////////////////////
		// determine number of blocks
		int n_blocks = (int)ceil((double)intended_sz/(sizeof(DATA_TYPE)*MAX_THREADS_PER_BLOCK));
		if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
		
		// run kernel
		conv_dfilter_nsum <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][imgs_ind], gpu_buffers[gpu_ind][deriv_above_ind], 
			gpu_buffers[gpu_ind][out_buffer_ind], dim_above, n_filters,
			conv_out_sz_x, n_channels, filter_sz, img_sz, PAD, intended_sz/sizeof(DATA_TYPE));
		
	}else{ // sum imgs
		status = cudnnConvolutionBackwardFilter(handle[gpu_ind], srcDesc[gpu_ind][imgs_ind], 
					gpu_buffers[gpu_ind][imgs_ind], 
					destDesc[gpu_ind][deriv_above_ind], 
					gpu_buffers[gpu_ind][deriv_above_ind], 
					convDesc[gpu_ind][out_buffer_ind], 
					gradDesc_filter[gpu_ind][out_buffer_ind], 
					gpu_buffers[gpu_ind][out_buffer_ind], 
					CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK
	}
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
