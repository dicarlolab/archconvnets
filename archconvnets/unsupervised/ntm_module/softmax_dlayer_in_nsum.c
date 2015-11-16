#define LAYER_OUT(A, B) layer_out[(A)*dim1 + B]
#define SMDLAYER(A, B, C, D) smdlayer[(A)*dim1*dim0*dim1 + \
	(B)*dim0*dim1 + (C)*dim1 + D]

static PyObject *softmax_dlayer_in_nsum(PyObject *self, PyObject *args){
    PyArrayObject *layer_out_in, *numpy_buffer_temp;
	float *layer_out, *smdlayer;
	
	if (!PyArg_ParseTuple(args, "iO!ii", &layer_out_ind, &PyTuple_Type, &layer_out_shape,
			&out_buffer_ind, &gpu_ind)) 
		return NULL;
		
	if(layer_out_ind >= N_BUFFERS || layer_out_ind < 0 || 
			out_buffer_ind >= N_BUFFERS || out_buffer_ind < 0){
		printf("buffer index incorrect, set_buffers().\n");
		return NULL;
	}
	
	int dim0 = PyArray_DIM(layer_out_in, 0);
	int dim1 = PyArray_DIM(layer_out_in, 1);
	
	layer_out = (float *) PyArray_DATA(layer_out_in);
	
	/////////////////////////  output buffer
	int dims[5];
	dims[0] = dim0;
	dims[1] = dim1;
	dims[2] = dim0;
	dims[3] = dim1;
	numpy_buffer_temp = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	smdlayer = (float *) PyArray_DATA(numpy_buffer_temp);
	
	for(int i = 0; i < dim0; i++){
		for(int j = 0; j < dim1; j++){
			SMDLAYER(i,j,i,j) = LAYER_OUT(i,j) * (1 - LAYER_OUT(i,j));
			for(int k = 0; k < dim1; k++){
				if(j != k)
					SMDLAYER(i,j,i,k) -= LAYER_OUT(i,j)*LAYER_OUT(i,k);
			}
		}
	}	
	return PyArray_Return(numpy_buffer_temp);
}
