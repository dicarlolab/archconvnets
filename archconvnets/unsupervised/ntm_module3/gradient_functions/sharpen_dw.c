#define W(A, B) w[(A)*dim1 + B]
#define DSDW_SZ (dim_above*dim0*dim1*sizeof(DATA_TYPE))
#define W_SZ buffer_sz[gpu_ind][w_ind]
#define WG_SUM wg[dim1]

__global__ void sharpen_dw_kernel(float * w, float * gamma, float * deriv_above, float * data_out, int dim0, int dim1, int dim_above){ 
	int i = blockIdx.x;
	int j = threadIdx.x / dim1;
	int k = threadIdx.x % dim1;

	extern __shared__ float shared_mem[];

	int a;
	float *wg = (float*)&shared_mem;
	float wg_sum2, g_wgm1, dsdw;
	
	// init
	if(k == 0){
		for(a = 0; a < dim_above; a++){
			data_out[a*dim0*dim1 + i*dim1 + j] = 0; // data_out[a,i,j]
		}
	}

	WG_SUM = 0;
	__syncthreads();
	if(k == 0){
		wg[j] = pow(W(i,j), gamma[i]);
		atomicAdd(&WG_SUM, wg[j]);
	}
	__syncthreads();
	wg_sum2 = WG_SUM * WG_SUM;

	g_wgm1 = gamma[i] * wg[j] / (W(i,j) * wg_sum2);
	
	// DSDW(i,k,j):
	if(k != j)
		dsdw = -g_wgm1 * wg[k];
	else
		dsdw = g_wgm1 * (WG_SUM - wg[j]);
	
	for(a = 0; a < dim_above; a++){
		//atomicAdd(&data_out[a,i,j],  dsdw[i,k,j] * deriv_above[a,i,k]);
		atomicAdd(&data_out[a*dim0*dim1 + i*dim1 + j],  dsdw * deriv_above[a*dim0*dim1 + i*dim1 + k]);
	}
}

// gamma: (dim0)
// w: (dim0,dim1)

// dsdw (dim0,dim1,dim0,dim1)

// deriv_above = (a, dim0,dim1)
//.........
// dsdw (dim0,dim1,dim1)
// deriv_above * dsdg = (a,dim0,dim1) .... sum dim1 (middle)...

static PyObject *sharpen_dw(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *w_shape, *gamma_shape, *deriv_above_shape;
	int w_ind, gamma_ind, out_buffer_ind, gpu_ind, deriv_above_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iO!iO!ii", &w_ind, &PyTuple_Type, &w_shape,
		&gamma_ind, &PyTuple_Type, &gamma_shape, &deriv_above_ind, &PyTuple_Type, &deriv_above_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(w_ind >= N_BUFFERS || w_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || 
			gamma_ind >= N_BUFFERS || gamma_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(GAMMA_SZ == 0 || W_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long dim_above = PyLong_AsLong(PyTuple_GetItem(deriv_above_shape,0));
	long dim0 = PyLong_AsLong(PyTuple_GetItem(w_shape,0));
	long dim1 = PyLong_AsLong(PyTuple_GetItem(w_shape,1));
	long dim0_gamma = PyLong_AsLong(PyTuple_GetItem(gamma_shape,0));
	
	if(dim0*dim1*sizeof(DATA_TYPE) != W_SZ || dim0*sizeof(DATA_TYPE) != GAMMA_SZ){
		printf("specified input sizes do not equal to stored gpu buffer. dot_cpu()\n");
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DSDW_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DSDW_SZ;
	}else if(DSDW_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}

	sharpen_dw_kernel <<< dim0, dim1*dim1, sizeof(float)*(dim1 + 1) >>> (gpu_buffers[gpu_ind][w_ind], gpu_buffers[gpu_ind][gamma_ind], 
		gpu_buffers[gpu_ind][deriv_above_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1, dim_above);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
