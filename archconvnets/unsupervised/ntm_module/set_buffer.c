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
	
	printf("7\n");
    cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	printf("5\n");
	data = (float *) PyArray_DATA(numpy_buffer_temp);
	printf(".\n");
	if(NUMPY_BUFFER == NULL){
		printf("3\n");
		//--------------------------------------
		// allocate filter, image, alpha, and beta tensors
		//----------------------------------------
		err = cudaMalloc((void**) &GPU_BUFFER, PyArray_NBYTES(numpy_buffer_temp)); MALLOC_ERR_CHECK
		
	}else{
		//-------------------------------------------
		// check to make sure inputs match the previously initialized buffer sizes
		//---------------------------------------------
		printf("%li\n", PyArray_NBYTES(numpy_buffer_temp));
		printf("%li\n", PyArray_NBYTES(NUMPY_BUFFER));
		if(PyArray_NBYTES(NUMPY_BUFFER) != PyArray_NBYTES(numpy_buffer_temp)){
				printf("---------------------------\ninput dimensions do not match the initial input dimensions on the first call to this function\n------------------\n");
				printf("4\n");
				return NULL;
		}
	}
	printf("8\n");
	//----------------------------------------
	// save input dimensions for error checking on subsequent calls
	//---------------------------------------
	NUMPY_BUFFER = numpy_buffer_temp;
	printf("2\n");
	//--------------------------------------
	// set image values
	//--------------------------------------
	err = cudaMemcpy(GPU_BUFFER, data, PyArray_NBYTES(NUMPY_BUFFER), cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	printf("1\n");
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	printf("%f %f\n", data[0],data[1]);
	//printf("%f %f\n", (float *)(PyArray_DATA(NUMPY_BUFFER[0]))[0], (float *)(PyArray_DATA(NUMPY_BUFFER[1]))[1]);
	
	Py_INCREF(Py_None);
	return Py_None;
}
