static PyObject * init_device(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind;
	
	if (!PyArg_ParseTuple(args, "i", &gpu_ind)) 
		return NULL;
    
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(device_init[gpu_ind] == 0){
		device_init[gpu_ind] = 1;
		cudaError_t err;
		cudnnStatus_t status;
		cublasStatus_t err_blas;
		
		status = cudnnCreatePoolingDescriptor(&poolingDesc);  ERR_CHECK
		status = cudnnSetPoolingDescriptor(poolingDesc, CUDNN_POOLING_MAX, POOL_WINDOW_SZ, POOL_WINDOW_SZ, POOL_STRIDE, POOL_STRIDE); ERR_CHECK
		
		/////////////////////////////////////////////////////////
		cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
		
		status = cudnnCreate(&handle[gpu_ind]);  ERR_CHECK
		err_blas = cublasCreate(&handle_blas[gpu_ind]); ERR_CHECK_BLAS
		
		for(int buffer_ind = 0; buffer_ind < N_BUFFERS; buffer_ind++){
			GPU_BUFFER = NULL;
			BUFFER_SZ = 0;
			
			//---------------------------------------
			// Create general Descriptors
			//---------------------------------------
			status = cudnnCreateTensor4dDescriptor(&srcDesc[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnCreateTensor4dDescriptor(&gradDesc_data[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnCreateTensor4dDescriptor(&destDesc[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnCreateFilterDescriptor(&filterDesc[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnCreateFilterDescriptor(&gradDesc_filter[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnCreateConvolutionDescriptor(&convDesc[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnCreateTensor4dDescriptor(&srcDiffDesc[gpu_ind][buffer_ind]);  ERR_CHECK
			status = cudnnCreateTensor4dDescriptor(&destDiffDesc[gpu_ind][buffer_ind]);  ERR_CHECK
		}
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
