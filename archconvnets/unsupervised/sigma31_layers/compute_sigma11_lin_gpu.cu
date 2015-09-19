#define S11_IND(A,B)((B) - (A) + offsets_c[A])
#define P_IND(A,B)((B) + (A)*(n_inds))

__global__ void kernel_sigma11_lin(int n_inds, int N_IMGS, float *sigma11_c, float *patches_c, IND_DTYPE *offsets_c, int ind_j_stride){
	int ind_i = blockIdx.x;
	int ind_j_start = threadIdx.x * ind_j_stride;
	
	if((ind_j_start) < ind_i) return;
	
	int img;
	
	float temp_sum = 0;
	
	int max_j = ind_j_start + ind_j_stride;
	if(max_j > n_inds){
		max_j = n_inds;
	}
	
	int ind_j;
	for(ind_j = ind_j_start; ind_j < max_j; ind_j++){
		temp_sum = 0;
		for(img = 0; img < N_IMGS; img++){
			temp_sum += patches_c[P_IND(img, ind_i)] * patches_c[P_IND(img, ind_j)];
		}
		atomicAdd(&sigma11_c[S11_IND(ind_i, ind_j)], temp_sum);
	}
}


static PyObject *compute_sigma11_lin_gpu(PyObject *self, PyObject *args){
	cudaError_t err;
	PyArrayObject *patches_in, *sigma11_in;
	
	int dims[14], i;
	
	float *patches, *patches_c, *sigma11, *sigma11_c;
	IND_DTYPE *offsets_c;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &patches_in)) return NULL;

	if (NULL == patches_in)  return NULL;
	
	int N_IMGS = PyArray_DIM(patches_in, 0);
	IND_DTYPE n_inds = PyArray_DIM(patches_in, 1);

	patches = (float *) patches_in -> data;
	
	IND_DTYPE n_pairs = 0.5*(n_inds-1)*n_inds + n_inds;
	
	dims[0] = n_pairs;
	sigma11_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	sigma11 = (float *) sigma11_in -> data;	
	
	/////////////////////////////////// offsets for indexing sigma11
	// (square coordinates to raveled, ex: i,j -> k)
	IND_DTYPE * offsets = NULL;
	
	offsets = (IND_DTYPE*)malloc(n_inds * sizeof(IND_DTYPE));
	if(NULL == offsets) return NULL;
	
	offsets[0] = 0;
	for(i = 1; i < n_inds; i++){
		offsets[i] = offsets[i-1] + n_inds - i + 1;
	}
	
	/////////////////////////////////////////// cuda mem
	cudaMalloc((void**) &patches_c, N_IMGS*n_inds * DATA_TYPE_SZ); CHECK_CUDA_ERR
	cudaMalloc((void**) &sigma11_c, n_pairs * DATA_TYPE_SZ); CHECK_CUDA_ERR
	cudaMalloc((void**) &offsets_c, n_inds * sizeof(IND_DTYPE)); CHECK_CUDA_ERR
	
	cudaMemcpy(patches_c, patches, N_IMGS*n_inds*DATA_TYPE_SZ, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(sigma11_c, sigma11, n_pairs*DATA_TYPE_SZ, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(offsets_c, offsets, n_inds * sizeof(IND_DTYPE), cudaMemcpyHostToDevice);  CHECK_CUDA_ERR	
	
	///////////////////////////
	dim3 grid_sz;
	grid_sz.x = n_inds;
	
	dim3 thread_sz;
	int ind_j_stride = 1;
	
	// can we index directly or do we need to stride?
	if(n_inds <= 1024)
		thread_sz.x = n_inds;
	else{
		thread_sz.x = 1024;
		ind_j_stride = ceil(n_inds/1024.0);
	}
	
	kernel_sigma11_lin <<< grid_sz, thread_sz >>> (n_inds, N_IMGS, sigma11_c, patches_c, offsets_c, ind_j_stride);
	
	cudaThreadSynchronize();
	
	CHECK_CUDA_ERR
	
	cudaMemcpy(sigma11, sigma11_c, n_pairs * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  CHECK_CUDA_ERR
	
	cudaFree(patches_c);
	cudaFree(sigma11_c);
	cudaFree(offsets_c);
	free(offsets);
	
	return PyArray_Return(sigma11_in);
}
