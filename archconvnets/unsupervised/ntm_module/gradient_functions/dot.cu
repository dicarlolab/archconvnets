#define GPU_BUFFER1 gpu_buffers[gpu_ind][buffer_ind1]
#define GPU_BUFFER2 gpu_buffers[gpu_ind][buffer_ind2]
#define GPU_BUFFER_OUT gpu_buffers[gpu_ind][out_buffer_ind]
#define BUFFER_SZ1 buffer_sz[gpu_ind][buffer_ind1]
#define BUFFER_SZ2 buffer_sz[gpu_ind][buffer_ind2]
#define OUT_BUFFER_SZ buffer_sz[gpu_ind][out_buffer_ind]

#define DATA_OUT(A, B) data_out[(A)*buffer2_dim2 + (B)]
#define DATA_OUT_IND(A, B) ((A)*buffer2_dim2 + (B))
#define DATA1(A, B) data1[(A)*buffer1_dim2 + (B)]
#define DATA1_IND(A, B) ((A)*buffer1_dim2 + (B))
#define DATA2(A, B) data2[(A)*buffer2_dim2 + (B)]
#define DATA2_IND(A, B) ((A)*buffer2_dim2 + (B))

#define DATA_OUT_SZ (buffer1_dim1*buffer2_dim2*sizeof(DATA_TYPE))
#define DATA_OUT_NUMEL (buffer1_dim1*buffer2_dim2)

__global__ void dot_kernel(float * data1, float * data2, float * data_out, int buffer1_dim1, int buffer1_dim2, int buffer2_dim1, 
			int buffer2_dim2, int data_out_numel){
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	int i,j;
	
	int min_duplicates_per_thread = (int)floor((double)data_out_numel / THREAD_CAPACITY);
	int n_additional_duplicates = data_out_numel % THREAD_CAPACITY;
	
	int n_duplicates = min_duplicates_per_thread;
	if(ind < n_additional_duplicates) n_duplicates++;
	
	int ind_g;
	unsigned data_out_ind, data1_ind, data2_ind;
	for(int dup = 0; dup < n_duplicates; dup++){
		ind_g = dup*THREAD_CAPACITY + ind;
		
		#ifdef DEBUG
		if(ind_g >= data_out_numel) assert(0); // out of bounds
		#endif
		
		i = ind_g / buffer2_dim2;
		j = ind_g % buffer2_dim2;
		
		data_out_ind = DATA_OUT_IND(i,j);
		data1_ind = DATA1_IND(i,0);
		data2_ind = DATA2_IND(0,j);
		
		data_out[data_out_ind] = 0;
		for(int k = 0; k < buffer1_dim2; k++){
			data_out[data_out_ind] += data1[data1_ind] * data2[data2_ind];
			
			data1_ind ++;
			data2_ind += buffer2_dim2;
		}
	}
}

static PyObject *dot(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind, buffer_ind1, buffer_ind2, out_buffer_ind;
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
		printf("specified input sizes do not equal to stored gpu buffer. dot_cpu()\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, DATA_OUT_SZ); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = DATA_OUT_SZ;
	}else if(DATA_OUT_SZ != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	// determine number of blocks
	int n_blocks = (int)ceil((double)DATA_OUT_NUMEL/MAX_THREADS_PER_BLOCK);
	if(n_blocks >= MAX_BLOCKS) n_blocks = MAX_BLOCKS;
	
	// run kernel
	dot_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (GPU_BUFFER1, GPU_BUFFER2, GPU_BUFFER_OUT, buffer1_dim1, buffer1_dim2, 
			buffer2_dim1, buffer2_dim2, DATA_OUT_NUMEL);
		
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
