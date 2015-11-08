#define GPU_BUFFER1 gpu_buffers[gpu_ind][buffer_ind1]
#define GPU_BUFFER2 gpu_buffers[gpu_ind][buffer_ind2]
#define BUFFER_SZ1 buffer_sz[gpu_ind][buffer_ind1]
#define BUFFER_SZ2 buffer_sz[gpu_ind][buffer_ind2]

#define DATA_OUT(A, B) data_out[(A)*buffer2_dim2 + (B)]
#define DATA1(A, B) data1[(A)*buffer1_dim2 + (B)]
#define DATA2(A, B) data2[(A)*buffer2_dim2 + (B)]

#define DATA_OUT_SZ (buffer1_dim1*buffer2_dim2*sizeof(DATA_TYPE))

static PyObject *dot_cpu(PyObject *self, PyObject *args){
    cudaError_t err;
	float *data1, *data2, *data_out;
	int gpu_ind, buffer_ind1, buffer_ind2;
	PyTupleObject *buffer_shape1, *buffer_shape2;
	
	if (!PyArg_ParseTuple(args, "iO!iO!i", &buffer_ind1, &PyTuple_Type, &buffer_shape1, &buffer_ind2, &PyTuple_Type, &buffer_shape2, &gpu_ind)) 
		return NULL;
        
	if(buffer_ind1 >= N_BUFFERS || buffer_ind1 < 0 || 
			buffer_ind2 >= N_BUFFERS || buffer_ind2 < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(BUFFER_SZ1 == 0 || BUFFER_SZ2 == 0){
		printf("buffer not initialized. use set_buffers()\n");
		return NULL;
	}
	
	// get sizes
	long buffer1_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)buffer_shape1,0));
	long buffer1_dim2 = PyLong_AsLong(PyTuple_GetItem((PyObject *)buffer_shape1,1));
	
	long buffer2_dim1 = PyLong_AsLong(PyTuple_GetItem((PyObject *)buffer_shape2,0));
	long buffer2_dim2 = PyLong_AsLong(PyTuple_GetItem((PyObject *)buffer_shape2,1));
	
	if(buffer1_dim1*buffer1_dim2*sizeof(DATA_TYPE) != BUFFER_SZ1 || buffer2_dim1*buffer2_dim2*sizeof(DATA_TYPE) != BUFFER_SZ2){
		printf("specified size not equal to stored gpu buffer. dot_cpu()\n");
		return NULL;
	}
	
    cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	MALLOC(data1, BUFFER_SZ1);
	MALLOC(data2, BUFFER_SZ2);
	MALLOC(data_out, DATA_OUT_SZ);
	
	err = cudaMemcpy(data1, GPU_BUFFER1, BUFFER_SZ1, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK
	err = cudaMemcpy(data2, GPU_BUFFER2, BUFFER_SZ2, cudaMemcpyDeviceToHost);  MALLOC_ERR_CHECK
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	memset(data_out, 0, DATA_OUT_SZ);
	
	clock_t begin, end;
	
	begin = clock();
	
	for(int i = 0; i < buffer1_dim1; i++){
		for(int k = 0; k < buffer1_dim2; k++){
			for(int j = 0; j < buffer2_dim2; j++){
				DATA_OUT(i,j) += DATA1(i,k) * DATA2(k,j);
			} // j
		} // k
	} // i
	
	end = clock();
	printf("%G seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
	
	free(data1);
	free(data2);
	free(data_out);
	
	Py_INCREF(Py_None);
	return Py_None;
}
