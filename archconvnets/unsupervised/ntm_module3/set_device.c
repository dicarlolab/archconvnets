static PyObject * set_device(PyObject *self, PyObject *args){
	cudaError_t err;
	int gpu_ind;
	
	if (!PyArg_ParseTuple(args, "i", &gpu_ind)) 
		return NULL;
    
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	cudaSetDevice(gpu_ind); CHECK_CUDA_ERR
	
	Py_INCREF(Py_None);
	return Py_None;
}
