#define DIDG_SZ (n_imgs*dim_above*dim0*sizeof(DATA_TYPE))

__global__ void interpolate_dinterp_gate_out_kernel(float * o_content, float * o_prev, float * deriv_above,
		float * out_data, int dim0, int dim1, int dim_above){ 
	int img = blockIdx.x / dim_above;
	int a = blockIdx.x % dim_above;
	int i = threadIdx.x;
	
	unsigned ind = img*dim_above*dim0 + a*dim0 + i;
	
	int ind_t = ind * dim1;
	int ind_t2 = img*dim0*dim1 + i*dim1;
	
	//out_data[a,i] = 0;
	out_data[ind] = 0;
	for(int j = 0; j < dim1; j++){
		//DIDG(i,j) = O_CONTENT(i,j) - O_PREV(i,j);
		//out_data[a,i] += (O_CONTENT(i,j) - O_PREV(i,j)) * deriv_above[a,i,j];
		out_data[ind] += (o_content[ind_t2 + j] - o_prev[ind_t2 + j]) * deriv_above[ind_t + j];
	}
}

// deriv_above: [a, dim0,dim1]
// didg: [dim0,dim1,dim0] -> [dim0,dim1]
// deriv_above * didg -> [a,dim0] (sum dim1)

static PyObject * interpolate_dinterp_gate_out(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *o_content_shape;
	int o_content_ind, o_prev_ind, out_buffer_ind, gpu_ind, deriv_above_ind;
	
	if (!PyArg_ParseTuple(args, "iO!iiii", &o_content_ind, &PyTuple_Type, &o_content_shape, &o_prev_ind, &deriv_above_ind,
			&out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(o_content_ind >= N_BUFFERS || o_content_ind < 0 ||
			o_prev_ind >= N_BUFFERS || o_prev_ind < 0 ||
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long n_imgs = PyLong_AsLong(PyTuple_GetItem(o_content_shape,0));
	long dim0 = PyLong_AsLong(PyTuple_GetItem(o_content_shape,1));
	long dim1 = PyLong_AsLong(PyTuple_GetItem(o_content_shape,2));
	
	long dim_above = buffer_sz[gpu_ind][deriv_above_ind] / buffer_sz[gpu_ind][o_content_ind];
	
	if(n_imgs*dim0*dim1*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][o_content_ind] ||
		n_imgs*dim0*dim1*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][o_prev_ind]){
		printf("specified input sizes do not equal to stored gpu buffer, %s\n", __FILE__);
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DIDG_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DIDG_SZ;
	}else if(DIDG_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	interpolate_dinterp_gate_out_kernel <<< n_imgs*dim_above, dim0 >>> (gpu_buffers[gpu_ind][o_content_ind], gpu_buffers[gpu_ind][o_prev_ind], 
		gpu_buffers[gpu_ind][deriv_above_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], dim0, dim1, dim_above);
	
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	Py_INCREF(Py_None);
	return Py_None;
}
