//-------------------------------------
// set_filter_buffer(): put filter data on GPU
// inputs: int filter_buff_ind, 
//          filters [n_filters, n_channels, filter_sz, filter_sz]
//			(ints): n_channels, filter_sz, n_filters

static PyObject *set_conv_buffer(PyObject *self, PyObject *args)  {
	int n_imgs_out;
	int n_filters_out;
	int conv_out_sz_x;
	int conv_out_sz_y;
	
	int conv_buff_ind, filter_buff_ind, img_buff_ind;
	
	if (!PyArg_ParseTuple(args, "iii", &conv_buff_ind, &filter_buff_ind, &img_buff_ind)) 
		return NULL;
	
	if(filter_buff_ind >= n_filter_buffers || conv_buff_ind >= n_conv_buffers || img_buff_ind >= n_img_buffers){
		printf("---------------\nrequested filter buffer ind greater than allocation. make sure to run init_buffers() first.\n----------\n");
		return NULL;
	}
	
	if(filterData_buffers[filter_buff_ind] == NULL || srcData_buffers[img_buff_ind] == NULL){
		printf("-----------\nfilters and img buffers must be filled before running this. run set_filter_buffer() and/or set_img_buffer() first\n----------\n");
		return NULL;
	}
	
	if(n_channels_imgs_buffers[img_buff_ind] != n_channels_filters_buffers[filter_buff_ind]){
		printf("---------\nfilter and image channel inds do not match up. check that buffer indices are correct.\n--------------------\n");
		return NULL;
	}
	
	cudaError_t err;
	cudnnStatus_t status;

	if(destData_buffers[conv_buff_ind] == NULL){
		//---------------------------------------
		// Set decriptor
		//---------------------------------------
		status = cudnnSetConvolutionDescriptor(convDesc_buffers[conv_buff_ind], srcDesc_buffers[img_buff_ind], filterDesc_buffers[filter_buff_ind], 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);  ERR_CHECK

		//---------------------------------------
		// Query output layout
		//---------------------------------------
		status = cudnnGetOutputTensor4dDim(convDesc_buffers[conv_buff_ind], CUDNN_CONVOLUTION_FWD, &n_imgs_out, &n_filters_out, &conv_out_sz_x, &conv_out_sz_y);    ERR_CHECK
		
		dims_buffers[conv_buff_ind] = n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x;
		
		//--------------------------------------
		// Set and allocate output tensor descriptor
		//----------------------------------------
		status = cudnnSetTensor4dDescriptor(destDesc_buffers[conv_buff_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs_out, n_filters_out, conv_out_sz_x, conv_out_sz_x); ERR_CHECK
		
		err = cudaMalloc((void**) &destData_buffers[conv_buff_ind], n_imgs_out*n_filters_out*conv_out_sz_x*conv_out_sz_x * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		//--------------------------------------
		// keep index variables
		//-----------------------------------
		conv_filter_ind[conv_buff_ind] = filter_buff_ind;
		conv_img_ind[conv_buff_ind] = img_buff_ind;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
