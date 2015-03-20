static PyObject *unpool(PyObject *self, PyObject *args){
	PyArrayObject *conv_output_in, *output_switches_x_in, *output_switches_y_in, *max_output_in;
	float *conv_output, *max_output;
	long *output_switches_x, *output_switches_y;
	int dims[10];
	int img_sz;
	
	if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &max_output_in, &PyArray_Type, &output_switches_x_in, &PyArray_Type, &output_switches_y_in, &img_sz)) 
		return NULL;
	
	if (NULL == max_output_in || NULL == output_switches_x_in || NULL == output_switches_y_in)  return NULL;
	
	max_output = (float *) max_output_in -> data;
	output_switches_x = (long *) output_switches_x_in -> data;
	output_switches_y = (long *) output_switches_y_in -> data;
	
	int n_filters = PyArray_DIM(max_output_in, 1);
	int n_imgs = PyArray_DIM(max_output_in, 0);
	int output_sz = PyArray_DIM(max_output_in, 2);
	
	dims[0] = n_imgs;
	dims[1] = n_filters;
	dims[2] = img_sz;
	dims[3] = img_sz;
	
	conv_output_in = (PyArrayObject *) PyArray_FromDims(4, dims, NPY_FLOAT);
	conv_output = (float *) conv_output_in -> data;
	
	int img, f, z1, z2;
	
	int img_sz_img_sz = img_sz*img_sz;
	int img_sz_img_sz_n_filters = img_sz*img_sz*n_filters;
	
	#define CONV_IND(A,B,C,D) ((D) + (C)*(img_sz) + (B)*(img_sz_img_sz) + (A)*(img_sz_img_sz_n_filters))
	
	int m_ind = 0;
	for(img = 0; img < n_imgs; img++){
		for(f = 0; f < n_filters; f++){
			for(z1 = 0; z1 < output_sz; z1++){
				for(z2 = 0; z2 < output_sz; z2++){
					conv_output[CONV_IND(img, f, output_switches_x[m_ind], output_switches_y[m_ind])] = max_output[m_ind];
					m_ind ++;
				} // z2
			} // z1
		} // f
	} // img
	
	return PyArray_Return(conv_output_in);
}
