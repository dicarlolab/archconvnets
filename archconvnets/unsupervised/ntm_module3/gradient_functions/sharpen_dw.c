#define DSDW_SZ (n_imgs*dim_above*dim0*dim1*sizeof(DATA_TYPE))
#define W_SZ buffer_sz[gpu_ind][w_ind]
#define WG_SUM wg[dim1]

__global__ void sharpen_dw_kernel(float * w, float * gamma, float * deriv_above, float * data_out, int dim0, int dim1, int dim_above){ 
	int img = blockIdx.x / dim0;
	int i = blockIdx.x % dim0;
	int j = threadIdx.x / dim1;
	int k = threadIdx.x % dim1;

	extern __shared__ float shared_mem[];

	int a;
	float *wg = (float*)&shared_mem; //last element in shared memory is for storing the sum
	float wg_sum2, g_wgm1, dsdw;
	
	int out_ind = img*dim_above*dim0*dim1 + i*dim1 + j;
	int gamma_ind = img*dim0 + i;
	int w_ind = img*dim0*dim1 + i*dim1 + j;
	int deriv_above_ind = img*dim_above*dim0*dim1 + i*dim1 + k;
	
	// init
	if(k == 0){
		for(a = 0; a < dim_above; a++){
			data_out[out_ind + a*dim0*dim1] = 0; // data_out[img,a,i,j]
		}
	}

	WG_SUM = 0;
	__syncthreads();
	if(k == 0){
		wg[j] = pow(w[w_ind], gamma[gamma_ind]);
		atomicAdd(&WG_SUM, wg[j]);
	}
	__syncthreads();
	wg_sum2 = WG_SUM * WG_SUM;

	g_wgm1 = gamma[gamma_ind] * wg[j] / (w[w_ind] * wg_sum2);
	
	// DSDW(i,k,j):
	if(k != j)
		dsdw = -g_wgm1 * wg[k];
	else
		dsdw = g_wgm1 * (WG_SUM - wg[j]);
	
	for(a = 0; a < dim_above; a++){
		//atomicAdd(&data_out[img, a,i,j],  dsdw[i,k,j] * deriv_above[a,i,k]);
		atomicAdd(&data_out[out_ind + a*dim0*dim1],  dsdw * deriv_above[deriv_above_ind + a*dim0*dim1]);
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
	PyObject *w_shape;
	int w_ind, gamma_ind, out_buffer_ind, gpu_ind, deriv_above_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iiii", &w_ind, &PyTuple_Type, &w_shape, &gamma_ind, &deriv_above_ind, &out_buffer_ind, &gpu_ind)) 
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
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(w_shape,0));
	long dim0 = PyLong_AsLong(PyTuple_GetItem(w_shape,1));
	long dim1 = PyLong_AsLong(PyTuple_GetItem(w_shape,2));
	
	long dim_above = buffer_sz[gpu_ind][deriv_above_ind] / buffer_sz[gpu_ind][w_ind];
	
	if(n_imgs*dim0*dim1*sizeof(DATA_TYPE) != W_SZ || n_imgs*dim0*sizeof(DATA_TYPE) != GAMMA_SZ){
		printf("specified input sizes do not equal to stored gpu buffer. %s\n", __FILE__);
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DSDW_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DSDW_SZ;
	}else if(DSDW_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size %s\n", __FILE__);
		return NULL;
	}

	sharpen_dw_kernel <<< n_imgs*dim0, dim1*dim1, sizeof(float)*(dim1 + 1) >>> (gpu_buffers[gpu_ind][w_ind], gpu_buffers[gpu_ind][gamma_ind], 
		gpu_buffers[gpu_ind][deriv_above_ind], gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1, dim_above);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
