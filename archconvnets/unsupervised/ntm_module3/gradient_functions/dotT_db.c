#define ADD_MEM_DADD_OUT_NUMEL (dim_above * n_imgs * C * mem_length)
#define ADD_MEM_DADD_OUT_SZ (ADD_MEM_DADD_OUT_NUMEL*sizeof(DATA_TYPE))

/*def add_mem_dadd_out(gw):
	temp = np.zeros((M, mem_length, C, mem_length),dtype='single')
	temp[:,range(mem_length),:,range(mem_length)] = gw.T
	return temp*/

// gw = (16, 6)  add_out = (16, 8)
// img, C, M    ....            img, C, mem_length

// deriv_above = (a, img, M, mem_length)
// deriv_above (a, img, M, mem_length) * gw (img, C, M) = [a, C, mem_length]

// batch a*img:
//		out[a,img] = gw[img] * deriv_above[a, img]


static PyObject *dotT_db(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, gw_ind, out_buffer_ind, deriv_above_ind, n_imgs, dim_above;
	PyObject *gw_shape, *add_out_shape;
	
	if (!PyArg_ParseTuple(args, "iO!O!iiiii", &gw_ind, &PyTuple_Type, &gw_shape, &PyTuple_Type, &add_out_shape, 
		&deriv_above_ind, &dim_above, &out_buffer_ind, &n_imgs, &gpu_ind)) 
		return NULL;
        
	if(out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || gw_ind >= N_BUFFERS || gw_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	int dim_offset = 0;
	if(n_imgs > 1)
		dim_offset ++;
	
	// get sizes
	long C = PyLong_AsLong(PyTuple_GetItem(gw_shape, dim_offset));
	long M = PyLong_AsLong(PyTuple_GetItem(gw_shape, 1 + dim_offset));
	
	long C2 = PyLong_AsLong(PyTuple_GetItem(add_out_shape, dim_offset));
	long mem_length = PyLong_AsLong(PyTuple_GetItem(add_out_shape, 1 + dim_offset));
	
	if(C != C2){
		printf("inner dot product dimensions do not match\n");
		return NULL;
	}
	
	if(n_imgs*C*M*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][gw_ind]){
		printf("specified input sizes do not equal to stored gpu buffer. %s\n", __FILE__);
		return NULL;
	}
	
	//cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, ADD_MEM_DADD_OUT_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = ADD_MEM_DADD_OUT_SZ;
	}else if(ADD_MEM_DADD_OUT_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	// deriv_above (a, img, M, mem_length) * gw (img, C, M) = [a, C, mem_length]

	// batch a*img:
	//		out[a,img] = gw[img] * deriv_above[a, img]
	
	cublasStatus_t err_blas;
	
	const float alpha = 1.0, beta = 0.0;
	
	for(int batch = 0; batch < dim_above; batch++){
		for(int img = 0; img < n_imgs; img++){
			// run kernel

			err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_N, mem_length, C, 
							M, &alpha,
                           gpu_buffers[gpu_ind][deriv_above_ind] + batch*n_imgs*M*mem_length + img*M*mem_length,
						   mem_length,
						   gpu_buffers[gpu_ind][gw_ind] + img*C*M, 
						   M, &beta, GPU_BUFFER_OUT + batch*n_imgs*C*mem_length + img*C*mem_length, mem_length);
			
			//err_blas = cublasSgemm(handle_blas[gpu_ind], CUBLAS_OP_N, CUBLAS_OP_N, buffer2_dim2, buffer1_dim1, buffer1_dim2, &alpha,
            //               GPU_BUFFER2, buffer2_dim2, GPU_BUFFER1, buffer1_dim2, &beta, GPU_BUFFER_OUT, buffer2_dim2);
			ERR_CHECK_BLAS
		}
	}
	#ifdef TIMING_DEBUG
		err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	#endif
	
	//cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
