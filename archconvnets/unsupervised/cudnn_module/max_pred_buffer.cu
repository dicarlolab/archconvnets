__global__ void kernel_max_pred_buffer(float * max_output3, float * pred, float * FL_pred, int N_C, int n_imgs, int n_filters, int conv_out_sz){
	int r = blockIdx.x;
	
	int cat = r / (n_filters*conv_out_sz*conv_out_sz);
	r = r % (n_filters*conv_out_sz*conv_out_sz);
	
	int f3 = r / (conv_out_sz*conv_out_sz);
	r = r % (conv_out_sz*conv_out_sz);
	
	int z1 = r / conv_out_sz;
	int z2 = r % conv_out_sz;
	
    int img = threadIdx.x;
	
	if(img == 0)
		FL_pred[blockIdx.x] = 0;
	__syncthreads();
	
	atomicAdd(&FL_pred[blockIdx.x], max_output3[img*(n_filters*conv_out_sz*conv_out_sz) + f3*(conv_out_sz*conv_out_sz) + z1*conv_out_sz + z2] * 
			pred[cat*n_imgs + img]);
	return;
}

// dFL === np.einsum(max_output3, range(4), pred, [4,0], [4,1,2,3])
// the gradient for FL filters [N_C x n3 x max_output_sz3 x max_output_sz3], given max_output3 [n_imgs, n3, max_output_sz3**2], and pred [N_C x n_imgs]
static PyObject *max_pred_buffer(PyObject *self, PyObject *args){
	cudaError_t err;
	cudnnStatus_t status;
	int gpu_ind, out_ind, max3_ind, pred_ind, stream_ind;
	
	if (!PyArg_ParseTuple(args, "iiiii", &max3_ind, &pred_ind, &out_ind, &stream_ind, &gpu_ind)) 
		return NULL;
	
	if(pred_ind >= N_BUFFERS || pred_ind < 0 || max3_ind >= N_BUFFERS || max3_ind < 0 || 
		out_ind >= N_BUFFERS || out_ind < 0){
		printf("invalid buffer index\n");
		return NULL;
	}
	
	if(gpu_ind < 0 || gpu_ind > N_GPUS){
		printf("invalid gpu index %i\n", gpu_ind);
		return NULL;
	}
	
	if(stream_ind < 0 || stream_ind > N_ALT_STREAMS){
		printf("invalid stream index %i\n", stream_ind);
		return NULL;
	}
	
	if(data_buffers[gpu_ind][max3_ind] == NULL || data_2d_buffers[gpu_ind][pred_ind] == NULL){
			printf("one or more buffers not initialized on this gpu\n");
			return NULL;
	}
	
	if(filter_flags[gpu_ind][max3_ind] == 1){
			printf("one or more buffers was not initialized correctly, filters when should be tensor or vice versa\n");
			return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	cudnnSetStream(handle, alt_streams[gpu_ind][stream_ind]);
	
	int n_imgs_out = data_dims[0][gpu_ind][max3_ind];
	int N_C = data_2d_dims[0][gpu_ind][pred_ind];
	int n_filters_out = data_dims[1][gpu_ind][max3_ind];
	int conv_out_sz = data_dims[2][gpu_ind][max3_ind];
	
	//--------------------------------------
	// Set and allocate output tensor descriptor
	//----------------------------------------
	if(data_buffers[gpu_ind][out_ind] == NULL){ // allocate output
		status = cudnnCreateTensor4dDescriptor(&desc_buffers[gpu_ind][out_ind]);  ERR_CHECK
		status = cudnnSetTensor4dDescriptor(desc_buffers[gpu_ind][out_ind], CUDNN_TENSOR_NCHW, dataType, N_C, n_filters_out, 
				conv_out_sz, conv_out_sz);  ERR_CHECK
		
		err = cudaMalloc((void**) &data_buffers[gpu_ind][out_ind], N_C*n_filters_out*conv_out_sz*conv_out_sz * DATA_TYPE_SZ); MALLOC_ERR_CHECK
		
		data_dims[0][gpu_ind][out_ind] = N_C;
		data_dims[1][gpu_ind][out_ind] = n_filters_out;
		data_dims[2][gpu_ind][out_ind] = conv_out_sz;
		data_dims[3][gpu_ind][out_ind] = conv_out_sz;
		
		filter_flags[gpu_ind][out_ind] = 0;

	//-------------------------------------------
	// check to make sure inputs match the previously initialized buffer sizes
	//---------------------------------------------
	}else if(data_dims[0][gpu_ind][out_ind] != N_C || data_dims[1][gpu_ind][out_ind] != n_filters_out || filter_flags[gpu_ind][out_ind] == 1 ||
			data_dims[2][gpu_ind][out_ind] != conv_out_sz || data_dims[3][gpu_ind][out_ind] != conv_out_sz){

				printf("---------------------------\ninput dimensions do not match the initial input dimensions on the first call to this function (%i %i %i %i), (%i %i %i)\n------------------\n", data_dims[0][gpu_ind][out_ind], data_dims[1][gpu_ind][out_ind],
				data_dims[2][gpu_ind][out_ind], data_dims[3][gpu_ind][out_ind], N_C, n_filters_out, conv_out_sz);
				return NULL;
	}
		
	/////////////////
	int thread_sz = N_C*n_filters_out*conv_out_sz*conv_out_sz;
	
	kernel_max_pred_buffer <<< thread_sz, n_imgs_out >>> (data_buffers[gpu_ind][max3_ind], data_2d_buffers[gpu_ind][pred_ind], 	data_buffers[gpu_ind][out_ind], N_C, n_imgs_out, n_filters_out, conv_out_sz);
	
	cudnnSetStream(handle, NULL);
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
