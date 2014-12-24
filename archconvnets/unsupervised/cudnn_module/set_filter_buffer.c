//-------------------------------------
// set_filter_buffer(): put filter data on GPU
// inputs: int filter_buff_ind, 
//          filters [n_filters, n_channels, filter_sz, filter_sz]

static PyObject *set_filter_buffer(PyObject *self, PyObject *args)  {
	PyArrayObject *filters_in;
	float *filters;
	int n_channels, filter_sz, n_filters, filter_buff_ind;
	
	if (!PyArg_ParseTuple(args, "iO!", &filter_buff_ind, &PyArray_Type, &filters_in)) 
		return NULL;
	if (NULL == filters)  return NULL;
	
	if(filter_buff_ind >= n_filter_buffers){
		printf("---------------\nrequested filter buffer ind greater than allocation. make sure to run init_buffers() first.\n----------\n", filter_buff_ind, n_filter_buffers);
		return NULL;
	}
	n_filters = PyArray_DIM(filters_in, 0);
	n_channels = PyArray_DIM(filters_in, 1);
	filter_sz = PyArray_DIM(filters_in, 2);
	
	filters = (float *) filters_in -> data;
	
	cudaError_t err;
	cudnnStatus_t status;

	if(filterData_buffers[filter_buff_ind] == NULL){
		//---------------------------------------
		// Set decriptor
		//---------------------------------------
		status = cudnnSetFilterDescriptor(filterDesc_buffers[filter_buff_ind], dataType, n_filters, n_channels, filter_sz, filter_sz);  ERR_CHECK

		//--------------------------------------
		// allocate filter, image, alpha, and beta tensors
		//----------------------------------------
		err = cudaMalloc((void**) &filterData_buffers[filter_buff_ind], n_filters*n_channels*filter_sz*filter_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		//----------------------------------------
		// save input dimensions for error checking on subsequent calls to conv()
		//---------------------------------------
		n_channels_filters_buffers[filter_buff_ind] = n_channels;
		filter_sz_buffers[filter_buff_ind] = filter_sz;
		n_filters_buffers[filter_buff_ind] = n_filters;
	}else{
		//-------------------------------------------
		// check to make sure inputs match the previously initialized buffer sizes
		//---------------------------------------------
		if(n_channels_filters_buffers[filter_buff_ind] != n_channels || filter_sz_buffers[filter_buff_ind] != filter_sz || n_filters_buffers[filter_buff_ind] != n_filters){
			printf("---------------------------\ninput dimensions [n_channels: %i, filter_sz: %i, n_filters: %i] do not match the initial input dimensions on the first call to this function [n_channels: %i, filter_sz: %i, n_filters: %i]. use conv() for general-purpose convolution.\n------------------\n", n_channels, filter_sz, n_filters, n_channels_filters_buffers[filter_buff_ind], filter_sz_buffers[filter_buff_ind], n_filters_buffers[filter_buff_ind]);
			return NULL;
		}
	}
	
	//--------------------------------------
	// set image values
	//--------------------------------------
	err = cudaMemcpy(filterData_buffers[filter_buff_ind], filters, n_filters*n_channels*filter_sz*filter_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	Py_INCREF(Py_None);
	return Py_None;
}
