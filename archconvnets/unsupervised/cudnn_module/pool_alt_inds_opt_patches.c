// conv_output: [n_imgs, n_filters, conv_output_sz, conv_output_sz]
// output_switches_x: [n_imgs, n_filters, output_sz, output_sz]
// imgs: [n_imgs, n_channels, img_sz, img_sz]
// int s [filter size]
static PyObject *max_pool_locs_alt_patches(PyObject *self, PyObject *args){
	PyArrayObject *conv_output_in, *output_switches_x_in, *output_switches_y_in, *output_in, *imgs_in, *pool_patches_in;
	PyObject * list;
	float *conv_output, *output, *imgs, *pool_patches;
	long *output_switches_x, *output_switches_y;
	int dims[1];
	int n_filters, n_imgs, output_sz, conv_output_sz, s, n_channels, img_sz;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!i", &PyArray_Type, &conv_output_in, &PyArray_Type, &output_switches_x_in, &PyArray_Type, &output_switches_y_in, &PyArray_Type, &imgs_in, &s)) 
		return NULL;
	
	if (NULL == conv_output_in || NULL == output_switches_x_in || NULL == output_switches_y_in)  return NULL;
	
	conv_output = (float *) conv_output_in -> data;
	imgs = (float *) imgs_in -> data;
	output_switches_x = (long *) output_switches_x_in -> data;
	output_switches_y = (long *) output_switches_y_in -> data;
	
	n_filters = PyArray_DIM(conv_output_in, 1);
	n_imgs = PyArray_DIM(conv_output_in, 0);
	output_sz = PyArray_DIM(output_switches_x_in, 2);
	conv_output_sz = PyArray_DIM(conv_output_in, 3);
	n_channels = PyArray_DIM(imgs_in, 1);
	img_sz = PyArray_DIM(imgs_in, 2);
	
	dims[0] = n_imgs*n_filters*output_sz*output_sz;
	output_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	output = (float *) output_in -> data;
	
	dims[0] = n_imgs*n_channels*s*s*n_filters*output_sz*output_sz;
	pool_patches_in = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT);
	pool_patches = (float *) pool_patches_in -> data;
	
	int output_sz_output_sz = output_sz * output_sz;
	int output_sz_output_sz_n_filters = output_sz * output_sz * n_filters;
	int conv_output_sz_conv_output_sz = conv_output_sz * conv_output_sz;
	int conv_output_sz_conv_output_sz_n_filters = conv_output_sz * conv_output_sz * n_filters;
	
	int output_sz_output_sz_n_filters_s_s_n_channels = output_sz_output_sz*n_filters*s*s*n_channels;
	int output_sz_output_sz_n_filters_s_s = output_sz_output_sz*n_filters*s*s;
	
	int img_sz_img_sz_n_channels = img_sz*img_sz*n_channels;
	int img_sz_img_sz = img_sz*img_sz;
	int output_sz_output_sz_n_filters_s = output_sz_output_sz*n_filters*s;
	
	int o_ind = 0, o_ind_L3 = 0, o_ind_L4 = 0;
	int s_ind = 0, s_ind_L1 = 0, s_ind_L2 = 0, s_ind_L3 = 0;
	int c_ind = 0, c_ind_L1 = 0, c_ind_L3 = 0;
	
	int pool_ind = 0;
	int x_loc, y_loc;
	int img_ind = 0, img_ind_L3 = 0, img_ind_L2 = 0, img_ind_L1 = 0;
	int p_ind_L4 = 0, p_ind_L3 = 0, p_ind_F = 0, p_ind_X = 0, p_ind_Y = 0, p_ind_Xo = 0;
	
	for(int img = 0; img < n_imgs; img++){
		o_ind_L3 = o_ind_L4;
		c_ind = c_ind_L3;
		p_ind_F = p_ind_L4;
		for(int filter = 0; filter < n_filters; filter++){
			o_ind = o_ind_L3;
			s_ind_L1 = s_ind_L2;
			p_ind_Xo = p_ind_F;
			for(int x = 0; x < output_sz; x++){
				for(int y = 0; y < output_sz; y++){
					s_ind = y + s_ind_L1;
					y_loc = output_switches_y[s_ind];
					x_loc = output_switches_x[s_ind];
					output[y + o_ind] = conv_output[y_loc + x_loc*conv_output_sz + c_ind];
					
					img_ind_L2 = img_ind_L3 + y_loc;
					p_ind_L3 = p_ind_Xo + y;
					for(int channel = 0; channel < n_channels; channel++){
						img_ind_L1 = x_loc*img_sz + img_ind_L2;
						p_ind_X = p_ind_L3;
						for(int x1 = 0; x1 < s; x1++){
							p_ind_Y = p_ind_X;
							for(int y1 = 0; y1 < s; y1++){
								pool_patches[p_ind_Y] = imgs[y1 + img_ind_L1];
								p_ind_Y += output_sz_output_sz_n_filters;
							} // y1
							img_ind_L1 += img_sz;
							p_ind_X += output_sz_output_sz_n_filters_s;
						} // x1
						img_ind_L2 += img_sz_img_sz;
						p_ind_L3 += output_sz_output_sz_n_filters_s_s;
					} // channel
				} // y
				o_ind += output_sz;
				s_ind_L1 += output_sz;
				p_ind_Xo += output_sz;
			} // x
			o_ind_L3 += output_sz_output_sz;
			s_ind_L2 += output_sz_output_sz;
			c_ind += conv_output_sz_conv_output_sz;
			p_ind_F += output_sz_output_sz;
		} // filter
		o_ind_L4 += output_sz_output_sz_n_filters;
		s_ind_L3 += output_sz_output_sz_n_filters;
		c_ind_L3 += conv_output_sz_conv_output_sz_n_filters;
		img_ind_L3 += img_sz_img_sz_n_channels;
		p_ind_L4 += output_sz_output_sz_n_filters_s_s_n_channels;
	} // img
	
	list = PyList_New(2);
	if(NULL == list) return NULL;
	
	if(-1 == PyList_SetItem(list, 0, PyArray_Return(output_in))) return NULL;
	if(-1 == PyList_SetItem(list, 1, PyArray_Return(pool_patches_in))) return NULL;
	
	return list;
}
