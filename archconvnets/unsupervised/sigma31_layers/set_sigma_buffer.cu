// inputs: sigma31, lex, GPU ind

static PyObject *set_sigma_buffer(PyObject *self, PyObject *args){
	PyArrayObject *sigma31_in;
	cudaError_t err;
	
	float *sigma31;
	
	int l, gpu_ind;
	
	if (!PyArg_ParseTuple(args, "O!ii", 
		&PyArray_Type, &sigma31_in, &l, &gpu_ind)) 
		return NULL;
	
	if (NULL == sigma31_in)  return NULL;

	if(l > N_LAYERS || l < 0 || gpu_ind >= N_GPUS){
		printf("invalid layer or gpu indices\n");
		return NULL;
	}
	
	if(sigma31s_c[gpu_ind][l] != 0){
		printf("sigma31 already initialized on this gpu for this layer\n");
		return NULL;
	}
	
	if(cudaSetDevice(gpu_ind) != cudaSuccess){
		err = cudaGetLastError();
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return NULL;
	}
	
	sigma31 = (float *) sigma31_in -> data;
	
	int sigma31_sz = PyArray_NBYTES(sigma31_in);
	
	//////////////////////////////////////////////////////////////////////////////// set global index size buffers
	n1s[l] = PyArray_DIM(sigma31_in, 1);
	n0s[l] = PyArray_DIM(sigma31_in, 2);
	s1s[l] = PyArray_DIM(sigma31_in, 3);
	n2s[l] = PyArray_DIM(sigma31_in, 5);
	s2s[l] = PyArray_DIM(sigma31_in, 6);
	n3s[l] = PyArray_DIM(sigma31_in, 8);
	s3s[l] = PyArray_DIM(sigma31_in, 9);
	max_output_sz3s[l] = PyArray_DIM(sigma31_in, 11);
	
	// indexing products
	max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l]*n3s[l]*s2s[l]*s2s[l]*n2s[l]*s1s[l]*s1s[l]*n0s[l]*n1s[l];
	max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l]*n3s[l]*s2s[l]*s2s[l]*n2s[l]*s1s[l]*s1s[l]*n0s[l];
	max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l]*n3s[l]*s2s[l]*s2s[l]*n2s[l]*s1s[l]*s1s[l];
	max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l]*n3s[l]*s2s[l]*s2s[l]*n2s[l]*s1s[l];
	max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l]*n3s[l]*s2s[l]*s2s[l]*n2s[l];
	max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l]*n3s[l]*s2s[l]*s2s[l];
	max_output_sz3_max_output_sz3_s3_s3_n3_s2s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l]*n3s[l]*s2s[l];
	max_output_sz3_max_output_sz3_s3_s3_n3s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l]*n3s[l];
	max_output_sz3_max_output_sz3_s3_s3s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l]*s3s[l];
	max_output_sz3_max_output_sz3_s3s[l] = max_output_sz3s[l]*max_output_sz3s[l]*s3s[l];
	max_output_sz3_max_output_sz3s[l] = max_output_sz3s[l]*max_output_sz3s[l];
	max_output_sz3s[l] = 0;
	z2b[l] = 1;
	
	// check which dims should be broadcasted
	if(n1s[l] != n1){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s[l] = 0;
	}
	if(n0s[l] != n0){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s[l] = 0;
	}
	if(s1s[l] != s1){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s[l] = 0;
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s[l] = 0;
	}
	if(n2s[l] != n2){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s[l] = 0;
	}
	if(s2s[l] != s2){
		max_output_sz3_max_output_sz3_s3_s3_n3s[l] = 0;
		max_output_sz3_max_output_sz3_s3_s3_n3_s2s[l] = 0;
	}
	if(s3s[l] != s3){
		max_output_sz3_max_output_sz3s[l] = 0;
		max_output_sz3_max_output_sz3_s3s[l] = 0;
	}
	if(n3s[l] != n3){
		max_output_sz3_max_output_sz3_s3_s3s[l] = 0;
	}
	if(max_output_sz3s[l] != max_output_sz3){
		max_output_sz3s[l] = 0;
		z2b[l] = 0;
	}
	
	/////////////////////////////////// cuda mem

	err = cudaMalloc((void**) &sigma31s_c[gpu_ind][l], sigma31_sz); MALLOC_ERR_CHECK
	
	err = cudaMemcpy(sigma31s_c[gpu_ind][l], sigma31, sigma31_sz, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	// check for error
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
