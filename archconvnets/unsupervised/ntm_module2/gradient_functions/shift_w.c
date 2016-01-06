#define N_SHIFTS 3
#define W_INTERP(A, B) w_interp[(A)*M + B]

__global__ void shift_w_kernel(float * shift_out, float * w_interp, float * out, int n_controllers, int M){ 
	int controller = threadIdx.x / n_controllers;
	int loc = threadIdx.x % n_controllers;
	
	int shift_out_ind = controller*N_SHIFTS;
	
	if(loc-1 >= 0)
		out[threadIdx.x] = shift_out[shift_out_ind]*W_INTERP(controller, loc-1);
	else
		out[threadIdx.x] = shift_out[shift_out_ind]*W_INTERP(controller, n_controllers-1);
	
	out[threadIdx.x] += shift_out[shift_out_ind+1]*W_INTERP(controller, loc);
	
	if(loc+1 < n_controllers)
		out[threadIdx.x] += shift_out[shift_out_ind+2]*W_INTERP(controller, loc+1);
	else
		out[threadIdx.x] += shift_out[shift_out_ind+2]*W_INTERP(controller, 0);
}

/*w_tilde = np.zeros_like(w_interp)
n_mem_slots = w_interp.shape[1]

for loc in range(n_mem_slots):
	w_tilde[:,loc] = shift_out[:,0]*w_interp[:,loc-1] + shift_out[:,1]*w_interp[:,loc] + \
			shift_out[:,2]*w_interp[:,(loc+1)%n_mem_slots]*/

static PyObject * shift_w(PyObject *self, PyObject *args){
	cudaError_t err;
	PyObject *w_interp_shape;
	int shift_out_ind, out_buffer_ind, gpu_ind, w_interp_ind;
	
	if (!PyArg_ParseTuple(args, "iiO!ii", &shift_out_ind, &w_interp_ind, &PyTuple_Type, &w_interp_shape, &out_buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(shift_out_ind >= N_BUFFERS || shift_out_ind < 0 || out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0 ||
			w_interp_ind >= N_BUFFERS || w_interp_ind < 0){ 
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	// get sizes
	long n_controllers = PyLong_AsLong(PyTuple_GetItem(w_interp_shape,0));
	long M = PyLong_AsLong(PyTuple_GetItem(w_interp_shape,1));
	
	if(n_controllers*M*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][w_interp_ind] || 
			n_controllers*N_SHIFTS*sizeof(DATA_TYPE) != buffer_sz[gpu_ind][shift_out_ind]){
		printf("specified input sizes do not equal to stored gpu buffer\n");
		return NULL;
	}
	
	if(OUT_BUFFER_SZ == 0){ // init output buffer
		err = cudaMalloc((void**) &GPU_BUFFER_OUT, buffer_sz[gpu_ind][w_interp_ind]); MALLOC_ERR_CHECK
		
		OUT_BUFFER_SZ = buffer_sz[gpu_ind][w_interp_ind];
	}else if(buffer_sz[gpu_ind][w_interp_ind] != OUT_BUFFER_SZ){ // does the output size match the buffer size?
		printf("output buffer size not allocated to correct size\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	shift_w_kernel <<< 1, n_controllers * M >>> (gpu_buffers[gpu_ind][shift_out_ind], gpu_buffers[gpu_ind][w_interp_ind],
		gpu_buffers[gpu_ind][out_buffer_ind], n_controllers, M);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
