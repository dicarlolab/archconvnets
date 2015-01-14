#define F1_IND(A,B,C,D)(D + (s1)*C + (s1*s1)*B + (s1*s1*n0)*A)
#define F2_IND(A,B,C,D)(D + (s2)*C + (s2*s2)*B + (s2*s2*n1)*A)
#define F3_IND(A,B,C,D)(D + (s3)*C + (s3*s3)*B + (s3*s3*n2)*A)
#define FL_IND(A,B,C,D)(D + (max_output_sz3)*C + (max_output_sz3*max_output_sz3)*B + (max_output_sz3*max_output_sz3*n3)*A)


__global__ void kernel_mem(float * sum_res, float * sigma31, float * F1, float * F2, float * F3, float * FL,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2, int max_output_sz3_max_output_sz3_s3_s3_n3_s2, int max_output_sz3_max_output_sz3_s3_s3_n3, int max_output_sz3_max_output_sz3_s3_s3,
		int max_output_sz3_max_output_sz3_s3, int max_output_sz3_max_output_sz3, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s, int max_output_sz3_max_output_sz3_s3_s3_n3_s2s, int max_output_sz3_max_output_sz3_s3_s3_n3s, int max_output_sz3_max_output_sz3_s3_s3s,
		int max_output_sz3_max_output_sz3_s3s, int max_output_sz3_max_output_sz3s, int n0, int n0s, int n1, int n1s, int n2, int n2s, int n3, int n3s,
		int max_output_sz3, int max_output_sz3s, int s1, int s1s, int s2, int s2s, int s3, int s3s, int N_C, int FL321_sz){
	int r, s31_ind, cat_ind;
	int cat_i, cat_j;
	int f1, f0;
	int s1x, s1y;
	int f2;
	int s2x, s2y;
	int f3;
	int s3x, s3y;
	int z1, z2;
	
	//----------------------------- init shared mem
	extern __shared__ float sum_res_shared[];
	__syncthreads();
	if(threadIdx.x < (N_C * N_C)){
		sum_res_shared[threadIdx.x] = 0;
	}
	
	int F_ind = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

	if(F_ind >= FL321_sz){
		return;
	}

	r = F_ind;

	cat_j = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1;
	
	f1 = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0;
	
	f0 = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1;
	
	s1x = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1;
	
	s1y = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2;
	
	f2 = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2;
	
	s2x = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2;
	
	s2y = r / max_output_sz3_max_output_sz3_s3_s3_n3;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3;
	
	f3 = r / max_output_sz3_max_output_sz3_s3_s3;
	r = r % max_output_sz3_max_output_sz3_s3_s3;
	
	s3x = r / max_output_sz3_max_output_sz3_s3;
	r = r % max_output_sz3_max_output_sz3_s3;
	
	s3y = r / max_output_sz3_max_output_sz3;
	r = r % max_output_sz3_max_output_sz3;
	
	z1 = r / max_output_sz3;
	z2 = r % max_output_sz3;
	
	// indices for FL321
	int f1s = 0, f0s = 0;
	int s1xs = 0, s1ys = 0;
	int f2s = 0;
	int s2xs = 0, s2ys = 0;
	int f3s = 0;
	int s3xs = 0, s3ys = 0;
	int z1s = 0, z2s = 0;
	
	// check which dims shouldn't be broadcasted
	if(n1s == n1){
		f1s = f1;
	}
	if(n0s == n0){
		f0s = f0;
	}
	if(s1s == s1){
		s1xs = s1x;
		s1ys = s1y;
	}
	if(n2s == n2){
		f2s = f2;
	}
	if(s2s == s2){
		s2xs = s2x;
		s2ys = s2y;
	}
	if(s3s == s3){
		s3xs = s3x;
		s3ys = s3y;
	}
	if(n3s == n3){
		f3s = f3;
	}
	if(max_output_sz3s == max_output_sz3){
		z1s = z1;
		z2s = z2;
	}
	
	s31_ind = S31_IND(0, f1s, f0s, s1xs, s1ys, f2s, s2xs, s2ys, f3s, s3xs, s3ys, z1s, z2s);
	cat_ind = cat_j*N_C;
	
	float F_prod = F1[F1_IND(f1, f0, s1x, s1y)] * F2[F2_IND(f2, f1, s2x, s2y)] * F3[F3_IND(f3, f2, s3x, s3y)] * FL[FL_IND(cat_j, f3, z1, z2)] ;
	
	for(cat_i = 0; cat_i < N_C; cat_i++){
		atomicAdd(&sum_res_shared[cat_ind], sigma31[s31_ind] * F_prod);
		
		cat_ind ++;
		s31_ind += max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s;
	}
	
	__syncthreads();
	if(threadIdx.x < (N_C * N_C)){
		atomicAdd(&sum_res[threadIdx.x], sum_res_shared[threadIdx.x]);
	}
	
}


// inputs: sigma31, FL321
//N_C * n1 * 3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3

static PyObject *einsum_cat_pairs_mem_gpu(PyObject *self, PyObject *args){
	PyArrayObject *sigma31_in, *FL_in, *F3_in, *F2_in, *F1_in;
	cudaError_t err;
	PyArrayObject *sum_res_in;
	
	/*if(cudaSetDevice(3) != cudaSuccess){
		err = cudaGetLastError();
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return NULL;
	}*/
	
	float *sigma31, *FL, *F3, *F2, *F1;
	float *sum_res;
	
	int dims[1];
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!", 
		&PyArray_Type, &sigma31_in, &PyArray_Type, &F1_in, &PyArray_Type, &F2_in, &PyArray_Type, &F3_in, &PyArray_Type, &FL_in)) 
		return NULL;

	if (NULL == sigma31_in || NULL == F1_in || NULL == F2_in || NULL == F3_in || NULL == FL_in)  return NULL;

	sigma31 = (float *) sigma31_in -> data;
	FL = (float *) FL_in -> data;
	F3 = (float *) F3_in -> data;
	F2 = (float *) F2_in -> data;
	F1 = (float *) F1_in -> data;
	
	//////////////////////// get dims
	
	// dims for FL321
	int N_C = PyArray_DIM(sigma31_in, 0);
	int n1 = PyArray_DIM(F1_in, 0);
	int n0 = PyArray_DIM(F1_in, 1);
	int s1 = PyArray_DIM(F1_in, 2);
	int n2 = PyArray_DIM(F2_in, 0);
	int s2 = PyArray_DIM(F2_in, 2);
	int n3 = PyArray_DIM(F3_in, 0);
	int s3 = PyArray_DIM(F3_in, 2);
	int max_output_sz3 = PyArray_DIM(FL_in, 2);

	// dims for sigma
	int n1s = PyArray_DIM(sigma31_in, 1);
	int n0s = PyArray_DIM(sigma31_in, 2);
	int s1s = PyArray_DIM(sigma31_in, 3);
	int n2s = PyArray_DIM(sigma31_in, 5);
	int s2s = PyArray_DIM(sigma31_in, 6);
	int n3s = PyArray_DIM(sigma31_in, 8);
	int s3s = PyArray_DIM(sigma31_in, 9);
	int max_output_sz3s = PyArray_DIM(sigma31_in, 11);

	int F1_sz = n1 * n0 * s1 * s1;
	int F2_sz = n1 * n2 * s2 * s2;
	int F3_sz = n2 * n3 * s3 * s3;
	int FL_sz = N_C * n3 * max_output_sz3 * max_output_sz3;
	int sigma31_sz = N_C * n1s * n0s * s1s * s1s * n2s * s2s * s2s * n3s * s3s * s3s * max_output_sz3s * max_output_sz3s;
	
	///////////////////////////////// allocate output mem
	dims[0] = N_C * N_C;
	
	sum_res_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	sum_res = (float *) sum_res_in -> data;
	
	/////////////////////////////////// cuda mem
	float * FL_c, * F3_c, * F2_c, * F1_c, * sigma31_c;
	float * sum_res_c;
	
	err = cudaMalloc((void**) &FL_c, FL_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &F3_c, F3_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &F2_c, F2_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &F1_c, F1_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &sigma31_c, sigma31_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	err = cudaMalloc((void**) &sum_res_c, N_C * N_C * DATA_TYPE_SZ); MALLOC_ERR_CHECK
	
	err = cudaMemcpy(FL_c, FL, FL_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(F3_c, F3, F3_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(F2_c, F2, F2_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(F1_c, F1, F1_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	err = cudaMemcpy(sigma31_c, sigma31, sigma31_sz * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	err = cudaMemcpy(sum_res_c, sum_res, N_C * N_C * DATA_TYPE_SZ, cudaMemcpyHostToDevice);  MALLOC_ERR_CHECK
	
	// indexing products
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*n0*n1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*n0;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2;
	int max_output_sz3_max_output_sz3_s3_s3_n3 = max_output_sz3*max_output_sz3*s3*s3*n3;
	int max_output_sz3_max_output_sz3_s3_s3 = max_output_sz3*max_output_sz3*s3*s3;
	int max_output_sz3_max_output_sz3_s3 = max_output_sz3*max_output_sz3*s3;
	int max_output_sz3_max_output_sz3 = max_output_sz3*max_output_sz3;
	
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s*s1s*s1s*n0s*n1s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s*s1s*s1s*n0s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s*s1s*s1s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s*s1s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s*n2s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s*s2s;
	int max_output_sz3_max_output_sz3_s3_s3_n3_s2s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s*s2s;
	int max_output_sz3_max_output_sz3_s3_s3_n3s = max_output_sz3s*max_output_sz3s*s3s*s3s*n3s;
	int max_output_sz3_max_output_sz3_s3_s3s = max_output_sz3s*max_output_sz3s*s3s*s3s;
	int max_output_sz3_max_output_sz3_s3s = max_output_sz3s*max_output_sz3s*s3s;
	int max_output_sz3_max_output_sz3s = max_output_sz3s*max_output_sz3s;
	
	
	//////////////////////////////////////////////////////////////////////////
	int FL321_sz = N_C * n1 * n0 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3;
	dim3 grid_size;
	grid_size.x = ceil(sqrt(FL321_sz / 1024.0));
	grid_size.y = grid_size.x;
	
	struct timeval tval_before, tval_after, tval_result;
	gettimeofday(&tval_before, NULL);
	
	kernel_mem<<<grid_size,1024, N_C * N_C * DATA_TYPE_SZ >>>(sum_res_c, sigma31_c, F1_c, F2_c, F3_c, FL_c, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2, max_output_sz3_max_output_sz3_s3_s3_n3_s2, max_output_sz3_max_output_sz3_s3_s3_n3, max_output_sz3_max_output_sz3_s3_s3,
		max_output_sz3_max_output_sz3_s3, max_output_sz3_max_output_sz3, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s, max_output_sz3_max_output_sz3_s3_s3_n3_s2s, max_output_sz3_max_output_sz3_s3_s3_n3s, max_output_sz3_max_output_sz3_s3_s3s,
		max_output_sz3_max_output_sz3_s3s, max_output_sz3_max_output_sz3s, n0, n0s, n1, n1s, n2, n2s, n3, n3s,
		max_output_sz3, max_output_sz3s, s1, s1s, s2, s2s, s3, s3s, N_C, FL321_sz);
	
	// make the host block until the device is finished with foo
	cudaThreadSynchronize();
	
	gettimeofday(&tval_after, NULL);
	timersub(&tval_after, &tval_before, &tval_result);
	printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
	
	// check for error
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return NULL;
	}
	
	err = cudaMemcpy(sum_res, sum_res_c, N_C * N_C * DATA_TYPE_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK
	
	cudaFree(FL_c);
	cudaFree(F3_c);
	cudaFree(F2_c);
	cudaFree(F1_c);
	cudaFree(sigma31_c);
	cudaFree(sum_res_c);
	
	return PyArray_Return(sum_res_in);
}
