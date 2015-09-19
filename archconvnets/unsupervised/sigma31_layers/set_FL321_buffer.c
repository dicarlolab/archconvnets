// inputs: FL321, GPU ind

static PyObject *set_FL321_buffer(PyObject *self, PyObject *args){
	PyArrayObject *FL321_in;
	cudaError_t err;
	
	float *FL321;
	
	int g; // gpu ind
	
	if (!PyArg_ParseTuple(args, "O!i", 
		&PyArray_Type, &FL321_in, &g)) 
		return NULL;
	
	if (NULL == FL321_in)  return NULL;

	if(g < 0 || g > N_GPUS){
		printf("invalid gpu index %i\n", g);
		return NULL;
	}
	
	cudaSetDevice(g); CHECK_CUDA_ERR
	
	unsigned long FL321_sz = PyArray_NBYTES(FL321_in);
	
	if(FL321s_c[g] != 0){
		cudaFree(FL321s_c[g]);
	}
	
	cudaMalloc((void**) &FL321s_c[g], FL321_sz); CHECK_CUDA_ERR
	
	FL321 = (float *) FL321_in -> data;
	
	N_Cs[g] = PyArray_DIM(FL321_in, 0);
	n_inds_FL321[g] = PyArray_DIM(FL321_in, 1);
	
	cudaMemcpy(FL321s_c[g], FL321, FL321_sz, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
