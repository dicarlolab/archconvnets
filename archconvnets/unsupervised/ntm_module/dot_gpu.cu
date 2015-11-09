#define GPU_BUFFER1 gpu_buffers[gpu_ind][buffer_ind1]
#define GPU_BUFFER2 gpu_buffers[gpu_ind][buffer_ind2]
#define GPU_BUFFER_OUT gpu_buffers[gpu_ind][out_buffer_ind]
#define BUFFER_SZ1 buffer_sz[gpu_ind][buffer_ind1]
#define BUFFER_SZ2 buffer_sz[gpu_ind][buffer_ind2]
#define OUT_BUFFER_SZ buffer_sz[gpu_ind][out_buffer_ind]

#define DATA_OUT(A, B) data_out[(A)*buffer2_dim2 + (B)]
#define DATA1(A, B) data1[(A)*buffer1_dim2 + (B)]
#define DATA2(A, B) data2[(A)*buffer2_dim2 + (B)]

#define DATA_OUT_NUMEL (buffer1_dim1*buffer2_dim2)

__global__ void dot_kernel(float * data1, float * data2, float * data_out, int buffer1_dim1, int buffer1_dim2, int buffer2_dim1, int buffer2_dim2){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= DATA_OUT_NUMEL) return;
	int i = ind / buffer2_dim2;
	int j = ind % buffer2_dim2;
	
	for(int k = 0; k < buffer1_dim2; k++){
		DATA_OUT(i,j) += DATA1(i,k) * DATA2(k,j);
	}
}

static PyObject *dot_gpu(PyObject *self, PyObject *args){
    cudaError_t err;
	int gpu_ind, buffer_ind1, buffer_ind2, out_buffer_ind;
	float * data_out;
	PyTupleObject *buffer_shape1, *buffer_shape2;
	
	if (!PyArg_ParseTuple(args, "iO!iO!ii", &buffer_ind1, &PyTuple_Type, &buffer_shape1, &buffer_ind2, 
			&PyTuple_Type, &buffer_shape2, &out_buffer_ind, &gpu_ind)) 
		return NULL;
        
	if(buffer_ind1 >= N_BUFFERS || buffer_ind1 < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 || 
			buffer_ind2 >= N_BUFFERS || buffer_ind2 < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(BUFFER_SZ1 == 0 || BUFFER_SZ2 == 0 || OUT_BUFFER_SZ == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long buffer1_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)buffer_shape1,0));
	long buffer1_dim2 = PyLong_AsLong(PyTuple_GetItem((PyObject *)buffer_shape1,1));
	
	long buffer2_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)buffer_shape2,0));
	long buffer2_dim2 = PyLong_AsLong(PyTuple_GetItem((PyObject *)buffer_shape2,1));
	
	if(buffer1_dim1*buffer1_dim2*sizeof(DATA_TYPE) != BUFFER_SZ1 || buffer2_dim1*buffer2_dim2*sizeof(DATA_TYPE) != BUFFER_SZ2){
		printf("specified input sizes do not equal to stored gpu buffer. dot_cpu()\n");
		return NULL;
	}
	
	if(DATA_OUT_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	MALLOC(data_out, DATA_OUT_SZ);
	
    cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	clock_t begin, end;
	
	dim3 thread_sz;
	thread_sz.x = buffer1_dim1;
	thread_sz.y = buffer2_dim2;
	
	int n_blocks = (int)ceil((double)DATA_OUT_NUMEL/MAX_THREADS_PER_BLOCK);
	
	begin = clock();
	dot_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (GPU_BUFFER1, GPU_BUFFER2, GPU_BUFFER_OUT, buffer1_dim1, buffer1_dim2, buffer2_dim1, buffer2_dim2);
	cudaDeviceSynchronize(); CHECK_CUDA_ERR
	end = clock();
	
	// copy result
	err = cudaMemcpy(data_out, GPU_BUFFER_OUT, DATA_OUT_SZ, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK
	
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	printf("%G seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
	
	printf("%f %f %f\n", DATA_OUT(0,1), DATA_OUT(1,0), DATA_OUT(1,1));
	
	free(data_out);
	
	Py_INCREF(Py_None);
	return Py_None;
}
