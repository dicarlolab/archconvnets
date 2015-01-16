// inputs: F1, F2, F3, FL, GPU ind

static PyObject *set_filter_buffers(PyObject *self, PyObject *args){
	PyArrayObject *F1_in, *F2_in, *F3_in, *FL_in;
	cudaError_t err;
	
	float *F1, *F2, *F3, *FL;
	
	int gpu_ind;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!i", 
		&PyArray_Type, &F1_in, &PyArray_Type, &F2_in, &PyArray_Type, &F3_in, &PyArray_Type, &FL_in, &gpu_ind)) 
		return NULL;
	
	if (NULL == F1_in || NULL == F2_in || NULL == F3_in || NULL == FL_in)  return NULL;

	if(gpu_ind >= N_GPUS){
		printf("invalid gpu index\n");
		return NULL;
	}
	
	if(cudaSetDevice(gpu_ind) != cudaSuccess){
		err = cudaGetLastError();
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return NULL;
	}
	
	FL = (float *) FL_in -> data;
	F3 = (float *) F3_in -> data;
	F2 = (float *) F2_in -> data;
	F1 = (float *) F1_in -> data;
	
	int F1_sz = PyArray_NBYTES(F1_in);
	int F2_sz = PyArray_NBYTES(F2_in);
	int F3_sz = PyArray_NBYTES(F3_in);
	int FL_sz = PyArray_NBYTES(FL_in);
	
	/////////////////////////////////// allocate cuda mem
	if(F1s_c[gpu_ind] == 0){
		err = cudaMalloc((void**) &F1s_c[gpu_ind], F1_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		err = cudaMalloc((void**) &F2s_c[gpu_ind], F2_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		err = cudaMalloc((void**) &F3s_c[gpu_ind], F3_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		err = cudaMalloc((void**) &FLs_c[gpu_ind], FL_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	}
	
	////////////////////////////////// set buffers
	err = cudaMemcpy(F1s_c[gpu_ind], F1, F1_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(F2s_c[gpu_ind], F2, F2_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(F3s_c[gpu_ind], F3, F3_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(FLs_c[gpu_ind], FL, FL_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	///////////////////////////////// set global dimensions used in the main einsum function
	N_C = PyArray_DIM(FL_in, 0);
	n1 = PyArray_DIM(F1_in, 0);
	n0 = PyArray_DIM(F1_in, 1);
	s1 = PyArray_DIM(F1_in, 2);
	n2 = PyArray_DIM(F2_in, 0);
	s2 = PyArray_DIM(F2_in, 2);
	n3 = PyArray_DIM(F3_in, 0);
	s3 = PyArray_DIM(F3_in, 2);
	max_output_sz3 = PyArray_DIM(FL_in, 2);
	
	// check for error
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
