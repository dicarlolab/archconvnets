#define S11_IND(A,B)((B) + (A)*(n_inds))
#define P_IND(A,B)((B) + (A)*(n_inds))

__global__ void kernel_sigma11(int n_inds, int N_IMGS, float *sigma11_c, float *patches_c, int ind_j_stride){
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


static PyObject *compute_sigma11_gpu(PyObject *self, PyObject *args){
	cudaError_t err;
	PyArrayObject *patches_in, *sigma11_in;
	
	int dims[14];
	
	float *patches, *patches_c, *sigma11, *sigma11_c;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &patches_in)) return NULL;

	if (NULL == patches_in)  return NULL;
	
	int N_IMGS = PyArray_DIM(patches_in, 0);
	int n_inds = PyArray_DIM(patches_in, 1);

	patches = (float *) patches_in -> data;
	
	dims[0] = n_inds;
	dims[1] = n_inds;
	
	sigma11_in = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT);
	sigma11 = (float *) sigma11_in -> data;	
	
	/////////////////////////////////////////// cuda mem
	cudaMalloc((void**) &patches_c, N_IMGS*n_inds * DATA_TYPE_SZ); CHECK_CUDA_ERR
	cudaMalloc((void**) &sigma11_c, n_inds*n_inds * DATA_TYPE_SZ); CHECK_CUDA_ERR
	
	cudaMemcpy(patches_c, patches, N_IMGS*n_inds*DATA_TYPE_SZ, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(sigma11_c, sigma11, n_inds*n_inds*DATA_TYPE_SZ, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	
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
	
	kernel_sigma11 <<< grid_sz, thread_sz >>> (n_inds, N_IMGS, sigma11_c, patches_c, ind_j_stride);
	
	cudaThreadSynchronize();
	
	CHECK_CUDA_ERR
	
	cudaMemcpy(sigma11, sigma11_c, n_inds*n_inds * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  CHECK_CUDA_ERR
	
	//symmetrize
	int ind_i, ind_j;
	for(ind_i = 0; ind_i < n_inds; ind_i++){
		for(ind_j = ind_i+1; ind_j < n_inds; ind_j++){
			sigma11[S11_IND(ind_j, ind_i)] = sigma11[S11_IND(ind_i, ind_j)];
		}
	}
	cudaFree(patches_c);
	cudaFree(sigma11_c);
	
	return PyArray_Return(sigma11_in);
}
