#define C_I(A,B,C,D,E)((E) + (D)*n_sets + (C)*n_sets_n_imgs + (B)*n_sets_n_imgs_conv_output_sz + (A)*n_sets_n_imgs_conv_output_sz_conv_output_sz)
#define O_I(A,B,C,D,E)((E) + (D)*n_sets + (C)*n_sets_n_imgs + (B)*n_sets_n_imgs_output_sz + (A)*n_sets_n_imgs_output_sz_output_sz)
#define S_I(A,B,C,D)((D) + (C)*n_imgs + (B)*n_imgs_output_sz + (A)*n_imgs_output_sz_output_sz)
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
	
	n_filters = PyArray_DIM(conv_output_in, 0);
	n_imgs = PyArray_DIM(conv_output_in, 3);
	output_sz = PyArray_DIM(output_switches_x_in, 1);
	conv_output_sz = PyArray_DIM(conv_output_in, 1);
	n_sets = PyArray_DIM(conv_output_in, 4);
	
	dims[0] = n_filters*output_sz*output_sz*n_imgs*n_sets;
	output_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	output = (float *) output_in -> data;
	
	int n_sets_n_imgs = n_sets*n_imgs;
	int n_sets_n_imgs_conv_output_sz = n_sets*n_imgs*conv_output_sz;
	int n_sets_n_imgs_conv_output_sz_conv_output_sz = n_sets*n_imgs*conv_output_sz*conv_output_sz;
	int n_sets_n_imgs_output_sz = n_sets*n_imgs*output_sz;
	int n_sets_n_imgs_output_sz_output_sz = n_sets*n_imgs*output_sz*output_sz;
	int n_imgs_output_sz_output_sz = n_imgs*output_sz*output_sz;
	int n_imgs_output_sz = n_imgs*output_sz;
	
	int output_L1, output_L2, output_L3, output_L4;
	int switch_ind, switch_ind_L2, switch_ind_L3, switch_ind_L4;
	int c_ind, c_ind_L4;
	int cp_sz = n_sets*sizeof(float);
	
	for(int filter = 0; filter < n_filters; filter++){
		output_L4 = filter*n_sets_n_imgs_output_sz_output_sz;
		switch_ind_L4 = filter*n_imgs_output_sz_output_sz;
		c_ind_L4 = filter*n_sets_n_imgs_conv_output_sz_conv_output_sz;
		for(int x = 0; x < output_sz; x++){
			output_L3 = x*n_sets_n_imgs_output_sz + output_L4;
			switch_ind_L3 = x*n_imgs_output_sz + switch_ind_L4;
			for(int y = 0; y < output_sz; y++){
				output_L2 = y*n_sets_n_imgs + output_L3;
				switch_ind_L2 = y*n_imgs + switch_ind_L3;
				for(int img = 0; img < n_imgs; img++){
					switch_ind = img + switch_ind_L2;
					memcpy(&output[img*n_sets + output_L2], 
						&conv_output[img*n_sets + output_switches_y[switch_ind]*n_sets_n_imgs + 
							output_switches_x[switch_ind]*n_sets_n_imgs_conv_output_sz + c_ind_L4], cp_sz);
					/*output_L1 = img*n_sets + output_L2;
					switch_ind = img + switch_ind_L2;
					c_ind = img*n_sets + output_switches_y[switch_ind]*n_sets_n_imgs + 
							output_switches_x[switch_ind]*n_sets_n_imgs_conv_output_sz + c_ind_L4;
					memcpy(&output[output_L1], &conv_output[c_ind], cp_sz);*/
				}
			}
		}
	}
	
	/*for(int set = 0; set < n_sets; set++){
		for(int filter = 0; filter < n_filters; filter++){
			for(int x = 0; x < output_sz; x++){
				for(int y = 0; y < output_sz; y++){
					for(int img = 0; img < n_imgs; img++){
						output[O_I(filter,x,y,img,set)] = conv_output[C_I(filter,output_switches_x[S_I(filter, x, y, img)],output_switches_y[S_I(filter, x, y, img)],img,set)];
					}
				}
			}
		}
	}*/
	
	return PyArray_Return(output_in);
}
