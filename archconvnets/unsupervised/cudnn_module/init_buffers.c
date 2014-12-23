char init = 0;
//-------------------------------------
// init_buffers(): initialize number of buffers for
// imgs, filters, convolutions (and outputs)


// inputs: n_img_buffers, n_filter_buffers, n_conv_buffers
static PyObject *init_buffers(PyObject *self, PyObject *args){
	cudnnStatus_t status;
	
	if(init){//if(n_img_buffers != 0 || n_filter_buffers != 0 || n_conv_buffers != 0){
		printf("-------------------\ninit_buffers() should only be called once per session\n-----------\n");
		return NULL;
	}
	n_img_buffers = N_BUFFERS;
	n_filter_buffers = N_BUFFERS;
	n_conv_buffers = N_BUFFERS;
	/*if (!PyArg_ParseTuple(args, "iii", &n_img_buffers, &n_filter_buffers, &n_conv_buffers)) 
		return NULL;
	if (0 == n_img_buffers || 0 == n_filter_buffers || 0 == n_conv_buffers)  return NULL;
	
	srcDesc_buffers = malloc(n_img_buffers * sizeof(cudnnTensor4dDescriptor_t));
	destDesc_buffers = malloc(n_conv_buffers * sizeof(cudnnTensor4dDescriptor_t));
	filterDesc_buffers = malloc(n_filter_buffers * sizeof(cudnnFilterDescriptor_t));
	convDesc_buffers = malloc(n_filter_buffers * sizeof(cudnnConvolutionDescriptor_t));
	
	srcData_buffers = malloc(n_img_buffers * sizeof(float*));
	destData_buffers = malloc(n_conv_buffers * sizeof(float*));
	filterData_buffers = malloc(n_img_buffers * sizeof(float*));

	n_channels_imgs_buffers = malloc(n_img_buffers * sizeof(int));
	img_sz_buffers = malloc(n_img_buffers * sizeof(int));
	n_imgs_buffers = malloc(n_img_buffers * sizeof(int));
	
	n_channels_filters_buffers = malloc(n_filter_buffers * sizeof(int));
	filter_sz_buffers = malloc(n_filter_buffers * sizeof(int));
	n_filters_buffers = malloc(n_filter_buffers * sizeof(int));
	
	dims_buffers = malloc(n_conv_buffers * sizeof(int));
	conv_filter_ind = malloc(n_conv_buffers * sizeof(int));
	conv_img_ind = malloc(n_conv_buffers * sizeof(int));*/
	
	//---------------------------------------
	// Create general Descriptors
	//---------------------------------------
	for(int filter = 0; filter < n_filter_buffers; filter++){
		status = cudnnCreateFilterDescriptor(&filterDesc_buffers[filter]);  ERR_CHECK
		filterData_buffers[filter] = NULL;
	}
	
	for(int img = 0; img < n_img_buffers; img++){
		status = cudnnCreateTensor4dDescriptor(&srcDesc_buffers[img]);  ERR_CHECK
		srcData_buffers[img] = NULL;
	}
	
	for(int conv = 0; conv < n_conv_buffers; conv++){
		status = cudnnCreateTensor4dDescriptor(&destDesc_buffers[conv]);  ERR_CHECK
		status = cudnnCreateConvolutionDescriptor(&convDesc_buffers[conv]);  ERR_CHECK
		destData_buffers[conv] = NULL;
	}
	
	init = 1;
	Py_INCREF(Py_None);
	return Py_None;
}
