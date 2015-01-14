#define S31_IND2(A,B,C,D,E,F,G,H,I,J,K,L,M)((M)*z2b + (L)*max_output_sz3s + (K)*max_output_sz3_max_output_sz3s + (J)*max_output_sz3_max_output_sz3_s3s + (I)*max_output_sz3_max_output_sz3_s3_s3s + \
	(H)*max_output_sz3_max_output_sz3_s3_s3_n3s + (G)*max_output_sz3_max_output_sz3_s3_s3_n3_s2s + (F)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s + (E)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s + \
	(D)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s + (C)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s + (B)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s + \
	(A)*max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s)
#define F1_IND(A,B,C,D)(D + (s1)*C + (s1*s1)*B + (s1*s1*n0)*A)
#define F2_IND(A,B,C,D)(D + (s2)*C + (s2*s2)*B + (s2*s2*n1)*A)
#define F3_IND(A,B,C,D)(D + (s3)*C + (s3*s3)*B + (s3*s3*n2)*A)
#define FL_IND(A,B,C,D)(D + (max_output_sz3)*C + (max_output_sz3*max_output_sz3)*B + (max_output_sz3*max_output_sz3*n3)*A)
	
__global__ void kernel(float * sum_res, float * sigma31, float * F1, float * F2, float * F3, float * FL,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2, int max_output_sz3_max_output_sz3_s3_s3_n3_s2, int max_output_sz3_max_output_sz3_s3_s3_n3, int max_output_sz3_max_output_sz3_s3_s3,
		int max_output_sz3_max_output_sz3_s3, int max_output_sz3_max_output_sz3, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s, int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s,
		int max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s, int max_output_sz3_max_output_sz3_s3_s3_n3_s2s, int max_output_sz3_max_output_sz3_s3_s3_n3s, int max_output_sz3_max_output_sz3_s3_s3s,
		int max_output_sz3_max_output_sz3_s3s, int max_output_sz3_max_output_sz3s, int z2b, int n0, int n0s, int n1, int n1s, int n2, int n2s, int n3, int n3s,
		int max_output_sz3, int max_output_sz3s, int s1, int s1s, int s2, int s2s, int s3, int s3s, int N_C, int f1_un, int f1_div, int f0_un, int f0_div, int f2_un, int f2_div, int f3_un, int f3_div, int s1x_un, int s1x_div,
		int s2x_un, int s2x_div, int s3x_un, int s3x_div, int s3y_un, int s3y_div){
	
	extern __shared__ float sum_res_shared[];
	if(threadIdx.x == 0){
		*sum_res_shared = 0;
	}
	__syncthreads();
	
	int r = blockIdx.x;
	int cat_j = r / N_C;
	int cat_i = r % N_C;
	
	int f1, f0;
	int s1x, s1y;
	int f2;
	int s2x, s2y;
	int f3;
	int s3x, s3y;
	int z1, z2;
	
	////////////////////////////////////////////////////////////////////////////////////////////////////
	// which dimensions have been unraveled across threads and that we should not loop over here?
	int *f1i = &f1, *f0i = &f0;
	int *s1xi = &s1x, *s1yi = &s1y;
	int *f2i = &f2;
	int *s2xi = &s2x, *s2yi = &s2y;
	int *f3i = &f3;
	int *s3xi = &s3x, *s3yi = &s3y;
	int *z1i = &z1, *z2i = &z2;
	
	int f1_sz = n1;
	int f0_sz = n0;
	int s1x_sz = s1;
	int s1y_sz = s1;
	int f2_sz = n2;
	int s2x_sz = s2;
	int s2y_sz = s2;
	int f3_sz = n3;
	int s3x_sz = s3;
	int s3y_sz = s3;
	int z1_sz = max_output_sz3;
	int z2_sz = max_output_sz3;
	
	
	r = threadIdx.x;
	
	if(f1_un){
		int f1c = r / f1_div;
		f1i = &f1c;
		r = r % f1_div;
		f1_sz = 1;
	}
	
	if(f0_un){
		int f0c = r / f0_div;
		f0i = &f0c;
		r = r % f0_div;
		f0_sz = 1;
	}
	
	if(f2_un){
		int f2c = r / f2_div;
		f2i = &f2c;
		r = r % f2_div;
		f2_sz = 1;
	}
	
	if(f3_un){
		int f3c = r / f3_div;
		f3i = &f3c;
		r = r % f3_div;
		f3_sz = 1;
	}
	
	if(s1x_un){
		int s1xc = r / s1x_div;
		s1xi = &s1xc;
		r = r % s1x_div;
		s1x_sz = 1;
	}
	
	if(s2x_un){
		int s2xc = r / s2x_div;
		s2xi = &s2xc;
		r = r % s2x_div;
		s2x_sz = 1;
	}
	
	if(s3x_un){
		int s3xc = r / s3x_div;
		s3xi = &s3xc;
		r = r % s3x_div;
		s3x_sz = 1;
	}
	
	if(s3y_un){
		int s3yc = r / s3y_div;
		s3yi = &s3yc;
		r = r % s3y_div;
		s3y_sz = 1;
	}
	
	float sum_res_local = 0;
	
	//N_C * n1 * 3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3
	for(f1 = 0; f1 < f1_sz; f1++){
		for(f0 = 0; f0 < f0_sz; f0++){
			for(s1x = 0; s1x < s1x_sz; s1x++){
				for(s1y = 0; s1y < s1y_sz; s1y++){
					for(f2 = 0; f2 < f2_sz; f2++){
						for(s2x = 0; s2x < s2x_sz; s2x++){
							for(s2y = 0; s2y < s2y_sz; s2y++){
								for(f3 = 0; f3 < f3_sz; f3++){
									for(s3x = 0; s3x < s3x_sz; s3x++){
										for(s3y = 0; s3y < s3y_sz; s3y++){
											for(z1 = 0; z1 < z1_sz; z1++){ 
												for(z2 = 0; z2 < z2_sz; z2++){
													sum_res_local += sigma31[S31_IND2(cat_i, *f1i, *f0i, *s1xi, s1y, *f2i, *s2xi, s2y, *f3i, *s3xi, *s3yi, z1, z2)] *
														F1[F1_IND(*f1i, *f0i, *s1xi, s1y)] * F2[F2_IND(*f2i, *f1i, *s2xi, s2y)] * F3[F3_IND(*f3i, *f2i, *s3xi, *s3yi)] * FL[FL_IND(cat_j, *f3i, z1, z2)];
												} // z2
											} // z1
										}
									} // s3x, s3y
								} // f3
							}
						} // s2x, s2y
					} // f2
				}
			} // s1x, s1y
		} // f0
	} // f1
	
	atomicAdd(&sum_res_shared[0], sum_res_local);
	
	__syncthreads();
	if(threadIdx.x == 0)
		sum_res[blockIdx.x] = *sum_res_shared;
	//atomicAdd(&sum_res[blockIdx.x], sum_res_local);
}


// inputs: sigma31, FL321
//N_C * n1 * 3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3

static PyObject *einsum_cat_pairs_gpu(PyObject *self, PyObject *args){
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
	int z2b = 1;
	
	// check which dims should be broadcasted
	if(n1s != n1){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s = 0;
	}
	if(n0s != n0){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s = 0;
	}
	if(s1s != s1){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s = 0;
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s = 0;
	}
	if(n2s != n2){
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s = 0;
	}
	if(s2s != s2){
		max_output_sz3_max_output_sz3_s3_s3_n3s = 0;
		max_output_sz3_max_output_sz3_s3_s3_n3_s2s = 0;
	}
	if(s3s != s3){
		max_output_sz3_max_output_sz3s = 0;
		max_output_sz3_max_output_sz3_s3s = 0;
	}
	if(n3s != n3){
		max_output_sz3_max_output_sz3_s3_s3s = 0;
	}
	if(max_output_sz3s != max_output_sz3){
		max_output_sz3s = 0;
		z2b = 0;
	}
	
	////////////////////////////////////////////////////
	// which dimensions do we unravel across threads? product of dim sizes should be <= 1024
	int f3_un = 0;
	int f2_un = 0;
	int f1_un = 0;
	int f0_un = 0;
	int s1x_un = 1;
	int s2x_un = 1;
	int s3x_un = 1;
	int s3y_un = 1;
	
	int f3_div = 1;
	int f2_div = 1;
	int f1_div = 1;
	int f0_div = 1;
	int s1x_div = 1;
	int s2x_div = 1;
	int s3x_div = 1;
	int s3y_div = 1;
	
	int thread_sz = 1;
	
	if(s3y_un){
		s3x_div *= s3;
		s2x_div *= s3;
		s1x_div *= s3;
		f3_div *= s3;
		f2_div *= s3;
		f0_div *= s3;
		f1_div *= s3;
		thread_sz *= s3;
	}
	
	if(s3x_un){
		s2x_div *= s3;
		s1x_div *= s3;
		f3_div *= s3;
		f2_div *= s3;
		f0_div *= s3;
		f1_div *= s3;
		thread_sz *= s3;
	}
	
	if(s2x_un){
		s1x_div *= s2;
		f3_div *= s2;
		f2_div *= s2;
		f0_div *= s2;
		f1_div *= s2;
		thread_sz *= s2;
	}
	
	if(s1x_un){
		f3_div *= s1;
		f2_div *= s1;
		f0_div *= s1;
		f1_div *= s1;
		thread_sz *= s1;
	}
	
	if(f3_un){
		f2_div *= n3;
		f0_div *= n3;
		f1_div *= n3;
		thread_sz *= n3;
	}
	
	if(f2_un){
		f0_div *= n2;
		f1_div *= n2;
		thread_sz *= n2;
	}
	
	if(f0_un){
		f1_div *= n0;
		thread_sz *= n0;
	}
	
	if(f1_un){
		thread_sz *= n1;
	}
	
	//////////////////////////////////////////////////////////////////////////
	
	//struct timeval tval_before, tval_after, tval_result;
	//gettimeofday(&tval_before, NULL);
	
	kernel <<< N_C * N_C, thread_sz, DATA_TYPE_SZ >>> (sum_res_c, sigma31_c, F1_c, F2_c, F3_c, FL_c, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2, max_output_sz3_max_output_sz3_s3_s3_n3_s2, max_output_sz3_max_output_sz3_s3_s3_n3, max_output_sz3_max_output_sz3_s3_s3,
		max_output_sz3_max_output_sz3_s3, max_output_sz3_max_output_sz3, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0_n1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_n0s,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1s, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2s,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2s, max_output_sz3_max_output_sz3_s3_s3_n3_s2s, max_output_sz3_max_output_sz3_s3_s3_n3s, max_output_sz3_max_output_sz3_s3_s3s,
		max_output_sz3_max_output_sz3_s3s, max_output_sz3_max_output_sz3s, z2b, n0, n0s, n1, n1s, n2, n2s, n3, n3s,
		max_output_sz3, max_output_sz3s, s1, s1s, s2, s2s, s3, s3s, N_C, f1_un, f1_div, f0_un, f0_div, f2_un, f2_div, f3_un, f3_div, s1x_un, s1x_div, s2x_un, s2x_div,
		s3x_un, s3x_div, s3y_un, s3y_div);
	
	// make the host block until the device is finished with foo
	cudaThreadSynchronize();
	
	/*gettimeofday(&tval_after, NULL);
	timersub(&tval_after, &tval_before, &tval_result);
	printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);*/
	
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
