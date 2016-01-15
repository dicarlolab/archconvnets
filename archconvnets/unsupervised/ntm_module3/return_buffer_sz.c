static PyObject *return_buffer_sz(PyObject *self, PyObject *args){
	int gpu_ind, buffer_ind;
	
	if (!PyArg_ParseTuple(args, "ii", &buffer_ind, &gpu_ind)) 
		return NULL;
    
	if(buffer_ind >= N_BUFFERS || buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	if(gpu_ind >= N_GPUS || gpu_ind < 0){
		printf("gpu index incorrect, set_buffers().\n");
		return NULL;
	}
	
	return Py_BuildValue("i", BUFFER_SZ / sizeof(DATA_TYPE));
}
