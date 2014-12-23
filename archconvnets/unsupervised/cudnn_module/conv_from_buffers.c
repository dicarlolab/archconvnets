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
	cudnnTensor4dDescriptor_t srcDescL = srcDesc_buffers[i];
	cudnnFilterDescriptor_t filterDescL = filterDesc_buffers[f];
	cudnnTensor4dDescriptor_t destDescL = destDesc_buffers[conv_buff_ind];
	cudnnConvolutionDescriptor_t convDescL = convDesc_buffers[conv_buff_ind];
	
	float * srcDataL = srcData_buffers[i];
	float * filterDataL = filterData_buffers[f];
	float * destDataL = destData_buffers[conv_buff_ind];

	//--------------------------------------
	// Convolution
	//--------------------------------------
	status = cudnnConvolutionForward(handle, srcDescL, srcDataL, filterDescL, filterDataL, convDescL, destDescL, destDataL, CUDNN_RESULT_NO_ACCUMULATE);  ERR_CHECK

	//--------------------------------------
	// Get output data
	//------------------------------------------
	err = (cudaError_t)cudaMemcpy(cout, destDataL, dims[0] * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK

	return PyArray_Return(vecout);
}
