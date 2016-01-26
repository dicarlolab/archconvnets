static PyObject *set_buffer(PyObject *self, PyObject *args){
    cudaError_t err;
	PyArrayObject *numpy_buffer_temp;
	float *data;
	int gpu_ind, buffer_ind;
	
	if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &numpy_buffer_temp, &buffer_ind, &gpu_ind)) 
		return NULL;
        
	if (NULL == numpy_buffer_temp)  return NULL;
    
	if(buffer_ind >= N_BUFFERS || buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
    //cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	data = (float *) PyArray_DATA(numpy_buffer_temp);
	if(BUFFER_SZ == 0){
		//--------------------------------------
		// allocate filter, image, alpha, and beta tensors
		//----------------------------------------
		err = cudaMalloc((void**) &GPU_BUFFER, PyArray_NBYTES(numpy_buffer_temp)); MALLOC_ERR_CHECK
		
		BUFFER_SZ = PyArray_NBYTES(numpy_buffer_temp);
	}else{
		//-------------------------------------------
		// check to make sure inputs match the previously initialized buffer sizes
		//---------------------------------------------
		if(BUFFER_SZ != PyArray_NBYTES(numpy_buffer_temp)){
				printf("---------------------------\ninput dimensions do not match the initial input dimensions on the first call to this function, %i: %li, %li\n------------------\n", buffer_ind, PyArray_NBYTES(numpy_buffer_temp), BUFFER_SZ);
				int buffer_ind2 = buffer_ind;
				for(buffer_ind = 0; buffer_ind < buffer_ind2+10; buffer_ind++){
					printf("%i %li\n", buffer_ind, BUFFER_SZ);
				}
				return NULL;
		}
	}
	
	//--------------------------------------
	// set image values
	//--------------------------------------
	err = cudaMemcpy(GPU_BUFFER, data, BUFFER_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
