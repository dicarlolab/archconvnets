// inputs: sigma11 GPU ind

static PyObject *set_sigma11_buffer(PyObject *self, PyObject *args){
	PyArrayObject *sigma11_in, *inds_in;
	cudaError_t err;
	
	float *sigma11;
	IND_DTYPE *inds;
	
	int g; // gpu ind
	
	if (!PyArg_ParseTuple(args, "O!O!i", 
		&PyArray_Type, &sigma11_in, &PyArray_Type, &inds_in, &g)) 
		return NULL;
	
	if (NULL == sigma11_in || NULL == inds_in)  return NULL;

	if(g < 0 || g > N_GPUS){
		printf("invalid gpu index %i\n", g);
		return NULL;
	}
	
	cudaSetDevice(g); CHECK_CUDA_ERR
	
	inds = (IND_DTYPE *) inds_in -> data;
	n_inds[g] = PyArray_DIM(inds_in, 0);
	
	IND_DTYPE n_pairs_c = 0.5*(n_inds[g]-1)*n_inds[g] + n_inds[g];
	
	IND_DTYPE n_pairs = PyArray_DIM(sigma11_in, 0);
	
	if(n_pairs_c != n_pairs){
		printf("sigma11 length not matching index length\n");
		return NULL;
	}
	
	unsigned long sigma11_sz = PyArray_NBYTES(sigma11_in);
	
	/////////////////////////////////// offsets for indexing sigma11
	// (square coordinates to raveled, ex: i,j -> k)
	IND_DTYPE * offsets = NULL;
	IND_DTYPE i;
	
	offsets = (IND_DTYPE*)malloc(n_inds[g] * sizeof(IND_DTYPE));
	if(NULL == offsets) return NULL;
	
	offsets[0] = 0;
	for(i = 1; i < n_inds[g]; i++){
		offsets[i] = offsets[i-1] + n_inds[g] - i + 1;
	}
	
	
	if(sigma11s_c[g] != 0){
		cudaFree(sigma11s_c[g]);
		cudaFree(offsets_c[g]);
		cudaFree(inds_c[g]);
	}
	
	cudaMalloc((void**) &sigma11s_c[g], sigma11_sz); CHECK_CUDA_ERR
	cudaMalloc((void**) &offsets_c[g], n_inds[g] * sizeof(IND_DTYPE)); CHECK_CUDA_ERR
	cudaMalloc((void**) &inds_c[g], n_inds[g] * sizeof(IND_DTYPE)); CHECK_CUDA_ERR
	
	sigma11 = (float *) sigma11_in -> data;
	
	cudaMemcpy(sigma11s_c[g], sigma11, sigma11_sz, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(inds_c[g], inds, n_inds[g] * sizeof(IND_DTYPE), cudaMemcpyHostToDevice); CHECK_CUDA_ERR
	cudaMemcpy(offsets_c[g], offsets, n_inds[g] * sizeof(IND_DTYPE), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR	
	
	free(offsets);
	Py_INCREF(Py_None);
	return Py_None;
}
