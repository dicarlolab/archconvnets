#define POOL_WINDOW_SZ 3
#define POOL_STRIDE 2
static PyObject *max_pool_locs(PyObject *self, PyObject *args){
	//cudaError_t err;
	PyObject * list;
	PyArrayObject *conv_output_in, *output_in, *output_switches_x_in, *output_switches_y_in;
	int pad;
	
	float * conv_output;
	int dims[4];
	
	if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &conv_output_in, &pad)) return NULL;
	
	conv_output = (float *) conv_output_in -> data;
	
	int n_imgs = PyArray_DIM(conv_output_in, 0);
	int n_filters = PyArray_DIM(conv_output_in, 1);
	int conv_sz = PyArray_DIM(conv_output_in, 2);
	
	int output_sz = ceil(((double)conv_sz - (double)POOL_WINDOW_SZ)/(double)POOL_STRIDE);
	
	dims[0] = n_imgs;
	dims[1] = n_filters;
	dims[2] = output_sz;
	dims[3] = output_sz;
	output_in = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	output_switches_x_in = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_INT);
	output_switches_y_in = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_INT);
	
	int x_loc, y_loc, filter, img;
	int x, y;
	
	x = 0;
	for(x_loc = 0; x_loc < (conv_sz-POOL_WINDOW_SZ); x_loc += POOL_STRIDE){
		y = 0;
		for(y_loc = 0; y_loc < (conv_sz-POOL_WINDOW_SZ); y_loc += POOL_STRIDE){
			for(filter = 0; filter < n_filters; filter++){
				for(img = 0; img < n_imgs; img++){
					
				} //img
			} //filter
			y++;
		} //y_loc
		x++;
	} // x_loc
	
	
	list = PyList_New(3);
	if(NULL == list) return NULL;
	
	if(-1 == PyList_SetItem(list, 0, PyArray_Return(output_in))) return NULL;
	if(-1 == PyList_SetItem(list, 1, PyArray_Return(output_switches_x_in))) return NULL;
	if(-1 == PyList_SetItem(list, 2, PyArray_Return(output_switches_y_in))) return NULL;
	
	return list;
}
