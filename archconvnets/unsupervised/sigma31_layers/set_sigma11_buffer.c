// inputs: sigma11 GPU ind

static PyObject *set_sigma11_buffer(PyObject *self, PyObject *args){
	PyArrayObject *sigma11_in;
	cudaError_t err;
	
	float *sigma11;
	
	int g; // gpu ind
	int warn;
	
	if (!PyArg_ParseTuple(args, "O!i", 
		&PyArray_Type, &sigma11_in, &g)) 
		return NULL;
	
	if (NULL == sigma11_in)  return NULL;

	if(g < 0 || g > N_GPUS){
		printf("invalid gpu index %i\n", g);
		return NULL;
	}
	
	cudaSetDevice(g); CHECK_CUDA_ERR
	
	unsigned long sigma11_sz = PyArray_NBYTES(sigma11_in);
	
	if(sigma11s_c[g] != 0){
		cudaFree(sigma11s_c[g]);
		cudaMalloc((void**) &sigma11s_c[g], sigma11_sz); CHECK_CUDA_ERR
	}else{
		/////////////////////////////////// cuda mem
		cudaMalloc((void**) &sigma11s_c[g], sigma11_sz); CHECK_CUDA_ERR
	}
	
	sigma11 = (float *) sigma11_in -> data;
	
	sigma11_len[g] = PyArray_DIM(sigma11_in, 0);
	
	cudaMemcpy(sigma11s_c[g], sigma11, sigma11_sz, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
