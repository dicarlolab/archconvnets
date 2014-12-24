//-------------------------------------
// set_img_buffer(): put image data on GPU
// inputs: int img_buff_ind, 
//          imgs [n_imgs, n_channels, img_sz, img_sz]

static PyObject *set_img_buffer(PyObject *self, PyObject *args)  {
	PyArrayObject *imgs_in;
	float *imgs;
	int n_channels, img_sz, n_imgs, img_buff_ind;
	
	if (!PyArg_ParseTuple(args, "iO!", &img_buff_ind, &PyArray_Type, &imgs_in)) 
		return NULL;
	if (NULL == imgs)  return NULL;
	
	if(img_buff_ind >= n_img_buffers){
		printf("---------------\nrequested img buffer ind greater than allocation. make sure to run init_buffers() first.\n----------\n", img_buff_ind, n_img_buffers);
		return NULL;
	}
	
	n_imgs = PyArray_DIM(imgs_in, 0);
	n_channels = PyArray_DIM(imgs_in, 1);
	img_sz = PyArray_DIM(imgs_in, 2);
	
	imgs = (float *) imgs_in -> data;
	
	cudaError_t err;
	cudnnStatus_t status;

	if(srcData_buffers[img_buff_ind] == NULL){
		//---------------------------------------
		// Set decriptor
		//---------------------------------------
		status = cudnnSetTensor4dDescriptor(srcDesc_buffers[img_buff_ind], CUDNN_TENSOR_NCHW, dataType, n_imgs, n_channels, img_sz, img_sz);  ERR_CHECK

		//--------------------------------------
		// allocate filter, image, alpha, and beta tensors
		//----------------------------------------
		err = cudaMalloc((void**) &srcData_buffers[img_buff_ind], n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		//----------------------------------------
		// save input dimensions for error checking on subsequent calls to conv()
		//---------------------------------------
		n_channels_imgs_buffers[img_buff_ind] = n_channels;
		img_sz_buffers[img_buff_ind] = img_sz;
		n_imgs_buffers[img_buff_ind] = n_imgs;
	}else{
		//-------------------------------------------
		// check to make sure inputs match the previously initialized buffer sizes
		//---------------------------------------------
		if(n_channels_imgs_buffers[img_buff_ind] != n_channels || img_sz_buffers[img_buff_ind] != img_sz || n_imgs_buffers[img_buff_ind] != n_imgs){
			printf("---------------------------\ninput dimensions [n_channels: %i, img_sz: %i, n_imgs: %i] do not match the initial input dimensions on the first call to this function [n_channels: %i, img_sz: %i, n_imgs: %i]. use conv() for general-purpose convolution.\n------------------\n", n_channels, img_sz, n_imgs, n_channels_imgs_buffers[img_buff_ind], img_sz_buffers[img_buff_ind], n_imgs_buffers[img_buff_ind]);
			return NULL;
		}
	}
	
	//--------------------------------------
	// set image values
	//--------------------------------------
	err = cudaMemcpy(srcData_buffers[img_buff_ind], imgs, n_imgs*n_channels*img_sz*img_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	Py_INCREF(Py_None);
	return Py_None;
}
