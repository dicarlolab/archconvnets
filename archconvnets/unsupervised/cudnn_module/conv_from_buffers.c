//-------------------------------------
// conv_from_buffers(): perform convolution based on 
// previously loaded buffers


// input: conv_buff_ind
// returns: conv_out [n_imgs, n_filters, conv_out_sz, conv_out_sz]
static PyObject *conv_from_buffers(PyObject *self, PyObject *args)  {
	PyArrayObject *vecout;
	float *cout;
	int conv_buff_ind;
	
	if (!PyArg_ParseTuple(args, "i", &conv_buff_ind)) 
		return NULL;
	if(destData_buffers[conv_buff_ind] == NULL){
		printf("-------------\nconv data structures not initialized. run set_conv_buffer()\n-------\n");
		return NULL;
	}
	
	int dims[1];
	dims[0] = dims_buffers[conv_buff_ind];
	cudaError_t err;

	cudnnStatus_t status;

	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	cout = (float *) vecout -> data;
	
	//-------------------
	// copy to temp vars
	//-----------------
	int i = conv_img_ind[conv_buff_ind];
	int f = conv_filter_ind[conv_buff_ind];
	
	//--------------------------------------
	// Convolution
	//--------------------------------------
	status = cudnnConvolutionForward(handle, srcDesc_buffers[i], srcData_buffers[i], filterDesc_buffers[f], filterData_buffers[f], 
			convDesc_buffers[conv_buff_ind], destDesc_buffers[conv_buff_ind], destData_buffers[conv_buff_ind], CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK

	//--------------------------------------
	// Get output data
	//------------------------------------------
	err = (cudaError_t)cudaMemcpy(cout, destData_buffers[conv_buff_ind], dims[0] * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK

	return PyArray_Return(vecout);
}
