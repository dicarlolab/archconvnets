// inputs: sigma31, lex, GPU ind

static PyObject *set_sigma_buffer(PyObject *self, PyObject *args){
	PyArrayObject *sigma31_in;
	cudaError_t err;
	
	float *sigma31;
	
	int l; // sigma buffer ind
	int g; // gpu ind
	
	if (!PyArg_ParseTuple(args, "O!ii", 
		&PyArray_Type, &sigma31_in, &l, &g)) 
		return NULL;
	
	if (NULL == sigma31_in)  return NULL;

	if(l < 0 || l > N_SIGMAS){
		printf("invalid sigma index %i\n", l);
		return NULL;
	}
	
	if(g < 0 || g > N_GPUS){
		printf("invalid gpu index %i\n", g);
		return NULL;
	}
	
	cudaSetDevice(g); CHECK_CUDA_ERR
	
	unsigned long sigma31_sz = PyArray_NBYTES(sigma31_in);
	
	if(sigma31s_c[g][l] != 0){
		printf("warning: sigma31 already initialized on this gpu for this layer\n");
		
		if(n1s[g][l] != PyArray_DIM(sigma31_in, 1) || n0s[g][l] != PyArray_DIM(sigma31_in, 2) || s1s[g][l] != PyArray_DIM(sigma31_in, 3) ||
			n2s[g][l] != PyArray_DIM(sigma31_in, 5) || s2s[g][l] != PyArray_DIM(sigma31_in, 6) || n3s[g][l] != PyArray_DIM(sigma31_in, 8) ||
			s3s[g][l] != PyArray_DIM(sigma31_in, 9) || max_output_sz3s[g][l] != PyArray_DIM(sigma31_in, 11)){
				printf("sigma dimensions must be the same as first initialized for sigma buffer ind %i on gpu %i\n", l, g);
				return NULL;
		}
	}else{
		//////////////////////////////////////////////////////////////////////////////// set global index size buffers
		n1s[g][l] = PyArray_DIM(sigma31_in, 1);
		n0s[g][l] = PyArray_DIM(sigma31_in, 2);
		s1s[g][l] = PyArray_DIM(sigma31_in, 3);
		n2s[g][l] = PyArray_DIM(sigma31_in, 5);
		s2s[g][l] = PyArray_DIM(sigma31_in, 6);
		n3s[g][l] = PyArray_DIM(sigma31_in, 8);
		s3s[g][l] = PyArray_DIM(sigma31_in, 9);
		max_output_sz3s[g][l] = PyArray_DIM(sigma31_in, 11);
	
		/////////////////////////////////// cuda mem
		cudaMalloc((void**) &sigma31s_c[g][l], sigma31_sz); CHECK_CUDA_ERR
	}
	
	sigma31 = (float *) sigma31_in -> data;
	
	cudaMemcpy(sigma31s_c[g][l], sigma31, sigma31_sz, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}

