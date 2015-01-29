#define C_IND(A,B,C,D)(D + (C)*conv_sz + (B)*conv_sz_conv_sz + (A)*conv_sz_conv_sz_n_filters)
#define O_IND(A,B,C,D)(D + (C)*output_sz + (B)*output_sz_output_sz + (A)*output_sz_output_sz_n_filters)

#define POOL_WINDOW_SZ 3
#define POOL_STRIDE 2
static PyObject *max_pool_locs(PyObject *self, PyObject *args){
	//cudaError_t err;
	PyObject * list;
	PyArrayObject *conv_output_in, *output_in, *output_switches_x_in, *output_switches_y_in;
	int pad;
	
	float * conv_output, *output;
	int dims[4];
	int *output_switches_x, *output_switches_y;
	
	if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &conv_output_in, &pad)) return NULL;
	
	if (NULL == conv_output_in)  return NULL;
	
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
	
	output = (float *) output_in -> data;
	output_switches_x = (int *) output_switches_x_in -> data;
	output_switches_y = (int *) output_switches_y_in -> data;
	
	int x_loc, y_loc, filter, img, offset_x, offset_y;
	int x, y;
	float max_px, t;
	
	int conv_sz_conv_sz = conv_sz*conv_sz;
	int conv_sz_conv_sz_n_filters = conv_sz*conv_sz*n_filters;
	int output_sz_output_sz = output_sz*output_sz;
	int output_sz_output_sz_n_filters = output_sz*output_sz*n_filters;
	int o_ind;
	
	printf("%f\n", conv_output[0]);
	printf("test %f %i %i %i %i %i %i\n", conv_output[C_IND(0,2,1,3)], n_imgs, n_filters, conv_sz,pad, PyArray_NBYTES(conv_output_in),sizeof(float));
	
	x = 0;
	for(x_loc = 0; x_loc < (conv_sz-POOL_WINDOW_SZ); x_loc += POOL_STRIDE){
		y = 0;
		for(y_loc = 0; y_loc < (conv_sz-POOL_WINDOW_SZ); y_loc += POOL_STRIDE){
			for(filter = 0; filter < n_filters; filter++){
				for(img = 0; img < n_imgs; img++){
					o_ind = O_IND(img, filter, x,y);
					max_px = -1000000;
					
					for(offset_x = 0; offset_x < POOL_WINDOW_SZ; offset_x++){
						for(offset_y = 0; offset_y < POOL_WINDOW_SZ; offset_y++){
							t = conv_output[C_IND(img, filter, x_loc+offset_x, y_loc+offset_y)];
							if(max_px < t){
								max_px = t;
								
								output_switches_x[o_ind] = x_loc+offset_x;
								output_switches_y[o_ind] = y_loc+offset_y;
							}
						} // offset_y
					} // offset_x
					
					if(output_switches_x[o_ind] < pad)
						output_switches_x[o_ind] = pad;
					else if((conv_sz - pad) <= output_switches_x[o_ind])
						output_switches_x[o_ind] = conv_sz - pad - 1;
					
					if(output_switches_y[o_ind] < pad)
						output_switches_y[o_ind] = pad;
					else if((conv_sz - pad) <= output_switches_y[o_ind])
						output_switches_y[o_ind] = conv_sz - pad - 1;
					
					
					output[o_ind] = conv_output[C_IND(img, filter, output_switches_x[o_ind],
						output_switches_y[o_ind])];
				} // img
			} // filter
			y++;
		} // y_loc
		x++;
	} // x_loc
	
	printf("test2\n");
	
	list = PyList_New(3);
	if(NULL == list) return NULL;
	printf("test2\n");
	if(-1 == PyList_SetItem(list, 0, PyArray_Return(output_in))) return NULL;
	if(-1 == PyList_SetItem(list, 1, PyArray_Return(output_switches_x_in))) return NULL;
	if(-1 == PyList_SetItem(list, 2, PyArray_Return(output_switches_y_in))) return NULL;
	printf("test2f\n");
	return list;
}
