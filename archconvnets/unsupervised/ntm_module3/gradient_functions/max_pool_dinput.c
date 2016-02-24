/*
# deriv_above [n_imgs, dim_above, n_filters, out_sz, out_sz]
# imgs [n_imgs, 3, in_sz, in_sz]

# dm_cpu = np.zeros((n_imgs, dim_above, n_filters, img_sz, img_sz), dtype='single')

for img in range(n_imgs):
	for filter in range(n_filters):
		for x in range(out_sz):
			for y in range(out_sz):
				window = imgs[img, filter, x*stride:x*stride+pool_width, y*stride:y*stride+pool_width] #.ravel()
				y_width = np.min([img_sz - y*stride, pool_width])
				
				window = window.ravel()
				out_cpu[img, filter, x,y] = np.max(window)
				
				max_loc = np.argmax(window)
				
				x_loc = max_loc / y_width
				y_loc = max_loc % y_width
				
				dm_cpu[img, :, filter, x*stride + x_loc, y*stride + y_loc] = deriv_above[img, :, filter, x,y]
*/

__global__ void max_pool_nsum(float * imgs, float * deriv_above, float * out, int dim_above, int n_filters,
		int out_sz, int img_sz, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	unsigned ind_g, out_ind, deriv_above_ind, img_ind;
	unsigned img, r, filter, x, y, x_loc, y_loc, x_loc_max, y_loc_max, a;
	
	float px_temp, px_max;
	
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		img = ind_g / (n_filters * out_sz * out_sz);
		r = ind_g % (n_filters * out_sz * out_sz);
		
		filter = r / (out_sz * out_sz);
		r = r % (out_sz * out_sz);
		
		x = r / out_sz;
		y = r % out_sz;
		//////// ind_g indexes deriv_above, now we find the index into our output for placing each element of deriv_above
		
		/////////// find max location
		img_ind = img*n_filters*img_sz*img_sz + filter*img_sz*img_sz + x*POOL_STRIDE*img_sz + y*POOL_STRIDE;
		
		x_loc_max=0; y_loc_max=0;
		//px_max = imgs[img, filter, x*POOL_STRIDE, y*POOL_STRIDE];
		px_max = imgs[img_ind];
		for(x_loc = 0; (x_loc < POOL_WINDOW_SZ) && ((x*POOL_STRIDE + x_loc) < img_sz); x_loc++){
			for(y_loc = 0; (y_loc < POOL_WINDOW_SZ) && ((y*POOL_STRIDE + y_loc) < img_sz); y_loc++){
				//px_temp = imgs[img, filter, x*POOL_STRIDE + x_loc, y*POOL_STRIDE + y_loc];
				px_temp = imgs[img_ind + x_loc*img_sz + y_loc];
				if(px_temp > px_max){
					x_loc_max = x_loc;
					y_loc_max = y_loc;
					px_max = px_temp;
				}
			}
		}
		
		out_ind = img*dim_above*n_filters*img_sz*img_sz + filter*img_sz*img_sz + (x*POOL_STRIDE + x_loc_max)*img_sz + y*POOL_STRIDE + y_loc_max;
		deriv_above_ind = img*dim_above*n_filters*out_sz*out_sz + filter*out_sz*out_sz + x*out_sz + y;
		
		for(a = 0; a < dim_above; a++){
			//out[img, a, filter, x*POOL_STRIDE + x_loc_max, y*POOL_STRIDE + y_loc_max] = deriv_above[img, a, filter, x, y]
			out[out_ind + a*n_filters*img_sz*img_sz] = deriv_above[deriv_above_ind + a*n_filters*out_sz*out_sz];
		}
		
	} // ind_g
}

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
		// since deriv_above has the dim_above dimension second (imgs are not contigiuous in mem, we cannot simply run cudnnPoolingBackward)
		
		/*unsigned deriv_above_offset, out_offset, img_offset, max_out_offset;
		
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
		}*/
		
		err = cudaMemset(gpu_buffers[gpu_ind][out_buffer_ind], 0, intended_sz);  MALLOC_ERR_CHECK
		/////////////////////////////////////
		// determine number of blocks
		int n_blocks = (int)ceil((double)intended_sz/(sizeof(DATA_TYPE)*MAX_THREADS_PER_BLOCK));
		if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
		
		// run kernel
		max_pool_nsum <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (gpu_buffers[gpu_ind][imgs_ind], gpu_buffers[gpu_ind][deriv_above_ind], 
			gpu_buffers[gpu_ind][out_buffer_ind], dim_above, n_channels, out_sz, img_sz, n_imgs*n_channels*out_sz*out_sz);
	}
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
