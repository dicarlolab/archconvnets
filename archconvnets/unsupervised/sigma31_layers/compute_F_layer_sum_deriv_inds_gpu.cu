#define P_IND(A,B)((B) + (A)*(n_inds))
#define S11_INDa(A,B)((B) - (A) + offsets_c[A])

#define F1S_IND(A, B, C, D)((D) + (C)*s1 + (B)*s1*s1 + (A)*s1*s1*3)
#define F2S_IND(A, B, C, D)((D) + (C)*s2 + (B)*s2*s2 + (A)*s2*s2*n1)
#define F3S_IND(A, B, C, D)((D) + (C)*s3 + (B)*s3*s3 + (A)*s3*s3*n2)
#define FLS_IND(A, B, C, D)((D) + (C)*max_output_sz3 + (B)*max_output_sz3_max_output_sz3 + (A)*max_output_sz3_max_output_sz3_n3)

__global__ void kernel_F_layer_sum_deriv_inds(float * F_sum, float * FL321, float * F_partial, float * sigma11, IND_DTYPE * inds, 
	IND_DTYPE * offsets_c,
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3,
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1,
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2, 
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2, IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3,
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3, IND_DTYPE max_output_sz3_max_output_sz3_s3, IND_DTYPE max_output_sz3_max_output_sz3,
	IND_DTYPE max_output_sz3, int layer_ind, IND_DTYPE n_inds, IND_DTYPE max_output_sz3_max_output_sz3_n3, int N_C,
	int s1, int s2, int s3, int n1, int n2, int n3, int ind_j_stride){
	
	int ind_i = blockIdx.x;
	int ind_j_start = threadIdx.x * ind_j_stride;
	
	if(ind_j_start >= n_inds) return;
	int max_j = ind_j_start + ind_j_stride;
	if(max_j > n_inds){
		max_j = n_inds;
	}
	
	////////////////////////////////////////////// unravel inds
		
	int f1_i = inds[ind_i] / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3;
	IND_DTYPE r = inds[ind_i] % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3;
	
	int channel_i = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1;
	
	int a1_x_i = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1;
	
	int a1_y_i = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2;
	
	int f2_i = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2;
	
	int a2_x_i = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2;
	
	int a2_y_i = r / max_output_sz3_max_output_sz3_s3_s3_n3;
	r = r % max_output_sz3_max_output_sz3_s3_s3_n3;
	
	int f3_i = r / max_output_sz3_max_output_sz3_s3_s3;
	r = r % max_output_sz3_max_output_sz3_s3_s3;
	
	int a3_x_i = r / max_output_sz3_max_output_sz3_s3;
	r = r % max_output_sz3_max_output_sz3_s3;
	
	int a3_y_i = r / max_output_sz3_max_output_sz3;
	r = r % max_output_sz3_max_output_sz3;
	
	int z1_i = r / (max_output_sz3);
	int z2_i = r % (max_output_sz3);
	
	IND_DTYPE F_sum_ind;
	if(layer_ind == 1){
		F_sum_ind = F1S_IND(f1_i, channel_i, a1_x_i, a1_y_i);
	}else if(layer_ind == 2){
		F_sum_ind = F2S_IND(f2_i, f1_i, a2_x_i, a2_y_i);
	}else if(layer_ind == 3){
		F_sum_ind = F3S_IND(f3_i, f2_i, a3_x_i, a3_y_i);
	}
	
	////////////////////////////////////////////// unravel inds
	int f1_j, channel_j, a1_x_j, a1_y_j, f2_j, a2_x_j, a2_y_j, f3_j, a3_x_j, a3_y_j, z1_j, z2_j;
	int ind_j, cat;
	float temp_sum = 0, sigma11_temp;
	
	for(ind_j = ind_j_start; ind_j < max_j; ind_j++){
		f1_j = inds[ind_j] / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3;
		r = inds[ind_j] % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3;
		
		channel_j = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1;
		r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1;
		
		a1_x_j = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1;
		r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1;
		
		a1_y_j = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2;
		r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2;
		
		f2_j = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2;
		r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2;
		
		a2_x_j = r / max_output_sz3_max_output_sz3_s3_s3_n3_s2;
		r = r % max_output_sz3_max_output_sz3_s3_s3_n3_s2;
		
		a2_y_j = r / max_output_sz3_max_output_sz3_s3_s3_n3;
		r = r % max_output_sz3_max_output_sz3_s3_s3_n3;
		
		f3_j = r / max_output_sz3_max_output_sz3_s3_s3;
		r = r % max_output_sz3_max_output_sz3_s3_s3;
		
		a3_x_j = r / max_output_sz3_max_output_sz3_s3;
		r = r % max_output_sz3_max_output_sz3_s3;
		
		a3_y_j = r / max_output_sz3_max_output_sz3;
		r = r % max_output_sz3_max_output_sz3;
		
		z1_j = r / (max_output_sz3);
		z2_j = r % (max_output_sz3);
		
		char matching = 0;
		if(layer_ind == 1 && f1_i == f1_j && channel_i == channel_j && a1_x_i == a1_x_j && a1_y_i == a1_y_j){
			matching = 1;
		}else if(layer_ind == 2 && f2_i == f2_j && f1_i == f1_j && a2_x_i == a2_x_j && a2_y_i == a2_y_j){
			matching = 1;
		}else if(layer_ind == 3 && f3_i == f3_j && f2_i == f2_j && a3_x_i == a3_x_j && a3_y_i == a3_y_j){
			matching = 1;
		}else if(layer_ind == 4 && f3_i == f3_j && z1_i == z1_j && z2_i == z2_j){
			matching = 1;
		}
		
		if(matching == 1){
			if(ind_i <= ind_j)
				sigma11_temp = sigma11[S11_INDa(ind_i, ind_j)];
			else
				sigma11_temp = sigma11[S11_INDa(ind_j, ind_i)];
			
			for(cat = 0; cat < N_C; cat++){
				if(layer_ind == 4){
					F_sum_ind = FLS_IND(cat, f3_i, z1_i, z2_i);
					atomicAdd(&F_sum[F_sum_ind], FL321[P_IND(cat, ind_i)] * F_partial[P_IND(cat, ind_j)] * sigma11_temp);
				}else{
					temp_sum += FL321[P_IND(cat, ind_i)] * F_partial[P_IND(cat, ind_j)] * sigma11_temp;
				}
			} // cat
		} // matching
	} //ind_j
	
	if(layer_ind != 4){
		atomicAdd(&F_sum[F_sum_ind], temp_sum);
	}
}

// layer_ind defines which layer to keep
static PyObject *compute_F_layer_sum_deriv_inds_gpu(PyObject *self, PyObject *args){
	cudaError_t err;
	PyArrayObject *F1_in, *F2_in, *F3_in, *FL_in;
	PyArrayObject *F_sum_in, *F_partial_in; // F_partial: FL321 sans the layer the deriv. is take wrt
	
	int layer_ind, gpu_ind;
	float *F_partial, *F_sum;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!ii",  &PyArray_Type, &F_partial_in, 
		&PyArray_Type, &F1_in, &PyArray_Type, &F2_in, &PyArray_Type, &F3_in, &PyArray_Type, &FL_in, 
		 &layer_ind, &gpu_ind)) return NULL;

	if (NULL == F_partial_in ||	NULL == F1_in || NULL == F2_in || NULL == F3_in || NULL == FL_in)  return NULL;

	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	if(sigma11s_c[gpu_ind] == 0 || inds_c[gpu_ind] == 0 || offsets_c[gpu_ind] == 0){
		printf("sigma11 buffer not set on gpu %i, call set_sigma11_buffer() first\n", gpu_ind);
		return NULL;
	}
	
	if(F_sum_c[gpu_ind][layer_ind] != 0 || F_partial_c[gpu_ind][layer_ind] != 0){
		printf("buffers not empty for layer %i on gpu %i, call F_layer_sum_deriv_inds_gpu_return() before calling this function again\n",layer_ind,gpu_ind);
		printf("N_C: %i, n_inds: %i\n", N_Cs[gpu_ind], n_inds[gpu_ind]);
		printf("F_sum_c: %i, F_partial_c: %i\n", F_sum_c[gpu_ind][layer_ind], F_partial_c[gpu_ind][layer_ind]);
		return NULL;
	}
	
	if(n_inds[gpu_ind] != n_inds_FL321[gpu_ind] && n_inds[gpu_ind] > 0){
		printf("number of indices for sigma11 not equal to the size of FL321.\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	F_partial = (float *) F_partial_in -> data;
	
	IND_DTYPE N_C = PyArray_DIM(FL_in, 0);
	IND_DTYPE max_output_sz3 = PyArray_DIM(FL_in, 2);
	IND_DTYPE n3 = PyArray_DIM(F3_in, 0);
	IND_DTYPE n2 = PyArray_DIM(F2_in, 0);
	IND_DTYPE n1 = PyArray_DIM(F1_in, 0);
	IND_DTYPE s1 = PyArray_DIM(F1_in, 2);
	IND_DTYPE s2 = PyArray_DIM(F2_in, 2);
	IND_DTYPE s3 = PyArray_DIM(F3_in, 2);
	
	if(N_C != N_Cs[gpu_ind]){
		printf("number of categories does not match number of categories from FL321 buffer. make sure set_FL321_buffer() was run with the correct inputs.\n");
		return NULL;
	}
	
	if(layer_ind == 1){ // F1 inds
		dims_F_sum[gpu_ind][layer_ind][0] = n1;
		dims_F_sum[gpu_ind][layer_ind][1] = 3;
		dims_F_sum[gpu_ind][layer_ind][2] = s1;
		dims_F_sum[gpu_ind][layer_ind][3] = s1;
	}else if(layer_ind == 2){
		dims_F_sum[gpu_ind][layer_ind][0] = n2;
		dims_F_sum[gpu_ind][layer_ind][1] = n1;
		dims_F_sum[gpu_ind][layer_ind][2] = s2;
		dims_F_sum[gpu_ind][layer_ind][3] = s2;
	}else if(layer_ind == 3){
		dims_F_sum[gpu_ind][layer_ind][0] = n3;
		dims_F_sum[gpu_ind][layer_ind][1] = n2;
		dims_F_sum[gpu_ind][layer_ind][2] = s3;
		dims_F_sum[gpu_ind][layer_ind][3] = s3;
	}else if(layer_ind == 4){
		dims_F_sum[gpu_ind][layer_ind][0] = N_C;
		dims_F_sum[gpu_ind][layer_ind][1] = n3;
		dims_F_sum[gpu_ind][layer_ind][2] = max_output_sz3;
		dims_F_sum[gpu_ind][layer_ind][3] = max_output_sz3;
	}else{
		printf("layer index (%i) not supported\n", layer_ind);
		return NULL;
	}
	
	F_sum_in = (PyArrayObject *) PyArray_FromDims(4, dims_F_sum[gpu_ind][layer_ind], NPY_FLOAT);
	F_sum = (float *) F_sum_in -> data;
	
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3 = max_output_sz3*max_output_sz3*s3*s3*n3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3 = max_output_sz3*max_output_sz3*s3*s3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3 = max_output_sz3*max_output_sz3*s3;
	IND_DTYPE max_output_sz3_max_output_sz3 = max_output_sz3*max_output_sz3;
	
	IND_DTYPE max_output_sz3_max_output_sz3_n3 = max_output_sz3*max_output_sz3*n3;
	
	//////////// cuda mem
	cudaMalloc((void**) &F_sum_c[gpu_ind][layer_ind], dims_F_sum[gpu_ind][layer_ind][0]*dims_F_sum[gpu_ind][layer_ind][1]*dims_F_sum[gpu_ind][layer_ind][2]*dims_F_sum[gpu_ind][layer_ind][3] * DATA_TYPE_SZ); CHECK_CUDA_ERR
	cudaMalloc((void**) &F_partial_c[gpu_ind][layer_ind], N_C*n_inds[gpu_ind] * DATA_TYPE_SZ); CHECK_CUDA_ERR
	
	cudaMemcpy(F_sum_c[gpu_ind][layer_ind], F_sum, dims_F_sum[gpu_ind][layer_ind][0]*dims_F_sum[gpu_ind][layer_ind][1]*dims_F_sum[gpu_ind][layer_ind][2]*dims_F_sum[gpu_ind][layer_ind][3]*DATA_TYPE_SZ, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	cudaMemcpy(F_partial_c[gpu_ind][layer_ind], F_partial, N_C*n_inds[gpu_ind]*DATA_TYPE_SZ, cudaMemcpyHostToDevice);  CHECK_CUDA_ERR
	
	//////////////
	// can we index directly or do we need to stride?
	int thread_sz;
	int ind_j_stride = 1;
	if(n_inds[gpu_ind] <= 1024)
		thread_sz = n_inds[gpu_ind];
	else{
		thread_sz = 1024;
		ind_j_stride = ceil(n_inds[gpu_ind]/1024.0);
	}
	
	///////////////////////////////
	kernel_F_layer_sum_deriv_inds <<<n_inds[gpu_ind],thread_sz>>>(F_sum_c[gpu_ind][layer_ind], FL321s_c[gpu_ind], F_partial_c[gpu_ind][layer_ind], sigma11s_c[gpu_ind], inds_c[gpu_ind], offsets_c[gpu_ind],
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1,
		max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2, max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2, 
		max_output_sz3_max_output_sz3_s3_s3_n3_s2, max_output_sz3_max_output_sz3_s3_s3_n3,
		max_output_sz3_max_output_sz3_s3_s3, max_output_sz3_max_output_sz3_s3, max_output_sz3_max_output_sz3,
		max_output_sz3, layer_ind, n_inds[gpu_ind], max_output_sz3_max_output_sz3_n3, N_C, s1, s2, s3, n1, n2, n3, ind_j_stride);
	
	Py_INCREF(Py_None);
	return Py_None;
}
