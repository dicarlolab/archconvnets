// conv_output: [n_imgs, n_sets, n_filters, conv_output_sz, conv_output_sz]
// output_switches_x: [n_imgs, n_filters, output_sz, output_sz]
static PyObject *max_pool_locs_alt(PyObject *self, PyObject *args){
	PyArrayObject *conv_output_in, *output_switches_x_in, *output_switches_y_in, *output_in;
	float *conv_output, *output;
	long *output_switches_x, *output_switches_y;
	int dims[1];
	int n_filters, n_imgs, output_sz, n_sets, conv_output_sz;
	
	if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &conv_output_in, &PyArray_Type, &output_switches_x_in, &PyArray_Type, &output_switches_y_in)) 
		return NULL;
	
	if (NULL == conv_output_in || NULL == output_switches_x_in || NULL == output_switches_y_in)  return NULL;
	
	conv_output = (float *) conv_output_in -> data;
	output_switches_x = (long *) output_switches_x_in -> data;
	output_switches_y = (long *) output_switches_y_in -> data;
	
	n_filters = PyArray_DIM(conv_output_in, 2);
	n_imgs = PyArray_DIM(conv_output_in, 0);
	output_sz = PyArray_DIM(output_switches_x_in, 2);
	conv_output_sz = PyArray_DIM(conv_output_in, 3);
	n_sets = PyArray_DIM(conv_output_in, 1);
	
	dims[0] = n_imgs*n_sets*n_filters*output_sz*output_sz;
	output_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	output = (float *) output_in -> data;
	
	int output_sz_output_sz = output_sz * output_sz;
	int output_sz_output_sz_n_filters = output_sz * output_sz * n_filters;
	int output_sz_output_sz_n_filters_n_sets = output_sz * output_sz * n_filters * n_sets;
	int conv_output_sz_conv_output_sz = conv_output_sz * conv_output_sz;
	int conv_output_sz_conv_output_sz_n_filters = conv_output_sz * conv_output_sz * n_filters;
	int conv_output_sz_conv_output_sz_n_filters_n_sets = conv_output_sz * conv_output_sz * n_filters * n_sets;
	int o_ind = 0, o_ind_L2 = 0, o_ind_L3 = 0, o_ind_L4 = 0;
	int s_ind = 0, s_ind_L1 = 0, s_ind_L2 = 0, s_ind_L3 = 0;
	int c_ind = 0, c_ind_L1 = 0, c_ind_L2 = 0, c_ind_L3 = 0;
	
	for(int img = 0; img < n_imgs; img++){
		o_ind_L3 = o_ind_L4;
		c_ind_L2 = c_ind_L3;
		for(int set = 0; set < n_sets; set++){
			o_ind_L2 = o_ind_L3;
			s_ind_L2 = s_ind_L3;
			c_ind = c_ind_L2;
			for(int filter = 0; filter < n_filters; filter++){
				o_ind = o_ind_L2;
				s_ind_L1 = s_ind_L2;
				for(int x = 0; x < output_sz; x++){
					for(int y = 0; y < output_sz; y++){
						s_ind = y + s_ind_L1;
						output[y + o_ind] = conv_output[output_switches_y[s_ind] + output_switches_x[s_ind]*conv_output_sz + c_ind];
					} // y
					o_ind += output_sz;
					s_ind_L1 += output_sz;
				} // x
				o_ind_L2 += output_sz_output_sz;
				s_ind_L2 += output_sz_output_sz;
				c_ind += conv_output_sz_conv_output_sz;
			} // filter
			o_ind_L3 += output_sz_output_sz_n_filters;
			c_ind_L2 += conv_output_sz_conv_output_sz_n_filters;
		} // set
		o_ind_L4 += output_sz_output_sz_n_filters_n_sets;
		s_ind_L3 += output_sz_output_sz_n_filters;
		c_ind_L3 += conv_output_sz_conv_output_sz_n_filters_n_sets;
	} // img
	
	return PyArray_Return(output_in);
}
