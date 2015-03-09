//N_C * n1 *3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3
#define P_IND(A,B)((B) + (A)*(n_inds))



// output_switches3_x, output_switches3_y, [n_imgs, n3, max_output_sz3, max_output_sz3]
// output_switches2_x, output_switches2_y, [n_imgs, n2, max_output_sz2, max_output_sz2]
// output_switches1_x, output_switches1_y, [n_imgs, n1, max_output_sz1, max_output_sz1]
// ints: s1, s2, s3
// labels [n_imgs]
// imgs: [n_imgs, 3, img_sz, img_sz] (float32)
// int: N_C
static PyObject *compute_patch_inds_addresses(PyObject *self, PyObject *args){
	PyObject * list;

	PyArrayObject *output_switches3_x_in, *output_switches3_y_in;
	PyArrayObject *output_switches2_x_in, *output_switches2_y_in;
	PyArrayObject *output_switches1_x_in, *output_switches1_y_in;
	PyArrayObject *x_in, *y_in, *channels_in;
	PyArrayObject *imgs_in, *inds_in;
	
	PyArrayObject *patches_in;
	
	int dims[14];
	int s1, s2, s3, N_C;
	long *output_switches3_x, *output_switches3_y;
	long *output_switches2_x, *output_switches2_y;
	long *output_switches1_x, *output_switches1_y;
	long *labels;
	IND_DTYPE *inds;
	float *imgs;
	float *patches;
	float *x, *y, *channels;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iiiO!iO!", 
		&PyArray_Type, &output_switches3_x_in, &PyArray_Type, &output_switches3_y_in,
		&PyArray_Type, &output_switches2_x_in, &PyArray_Type, &output_switches2_y_in,
		&PyArray_Type, &output_switches1_x_in, &PyArray_Type, &output_switches1_y_in,
		&s1, &s2, &s3, &PyArray_Type, &imgs_in, &N_C, &PyArray_Type, &inds_in)) 
		return NULL;

	if (NULL == output_switches3_x_in || NULL == output_switches3_y_in ||
		NULL == output_switches2_x_in || NULL == output_switches2_y_in ||
		NULL == output_switches1_x_in || NULL == output_switches1_y_in ||
		NULL == imgs_in || NULL == inds_in)  return NULL;

	
	imgs = (float *) imgs_in -> data;
	inds = (IND_DTYPE *) inds_in -> data;
	output_switches3_x = (long *) output_switches3_x_in -> data;
	output_switches3_y = (long *) output_switches3_y_in -> data;

	output_switches2_x = (long *) output_switches2_x_in -> data;
	output_switches2_y = (long *) output_switches2_y_in -> data;

	output_switches1_x = (long *) output_switches1_x_in -> data;
	output_switches1_y = (long *) output_switches1_y_in -> data;

	IND_DTYPE N_IMGS = PyArray_DIM(imgs_in, 0);
	IND_DTYPE img_sz = PyArray_DIM(imgs_in, 2);
	IND_DTYPE max_output_sz3 = PyArray_DIM(output_switches3_x_in, 2);
	IND_DTYPE max_output_sz2 = PyArray_DIM(output_switches2_x_in, 2);
	IND_DTYPE max_output_sz1 = PyArray_DIM(output_switches1_x_in, 2);
	IND_DTYPE n3 = PyArray_DIM(output_switches3_x_in, 1);
	IND_DTYPE n2 = PyArray_DIM(output_switches2_x_in, 1);
	IND_DTYPE n1 = PyArray_DIM(output_switches1_x_in, 1);
	IND_DTYPE n_inds = PyArray_DIM(inds_in, 0);
	
	dims[0] = N_IMGS;
	dims[1] = n_inds;
	
	patches_in = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT);
	patches = (float *) patches_in -> data;
	
	channels_in = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT);
	channels = (float *) channels_in -> data;
	
	x_in = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT);
	x = (float *) x_in -> data;
	
	y_in = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT);
	y = (float *) y_in -> data;
	
	int f1, channel, a1_x, a1_y, f2, a2_x, a2_y, f3, a3_x, a3_y, z1, z2, cat, img, ind;
	int a3_x_global, a3_y_global, a2_x_global, a2_y_global, a1_x_global, a1_y_global;
	
	IND_DTYPE r;
	
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3_n1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*3*n1;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1_3 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1*3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1*s1;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2_s1 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2*s1;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2_n2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2*n2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2*s2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3_s2 = max_output_sz3*max_output_sz3*s3*s3*n3*s2;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3_n3 = max_output_sz3*max_output_sz3*s3*s3*n3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3_s3 = max_output_sz3*max_output_sz3*s3*s3;
	IND_DTYPE max_output_sz3_max_output_sz3_s3 = max_output_sz3*max_output_sz3*s3;
	IND_DTYPE max_output_sz3_max_output_sz3 = max_output_sz3*max_output_sz3;
	
	IND_DTYPE max_output_sz3_max_output_sz3_n3 = max_output_sz3*max_output_sz3*n3;
	
	IND_DTYPE max_output_sz2_max_output_sz2 = max_output_sz2*max_output_sz2;
	IND_DTYPE max_output_sz2_max_output_sz2_n2 = max_output_sz2*max_output_sz2*n2;
	
	IND_DTYPE max_output_sz1_max_output_sz1 = max_output_sz1*max_output_sz1;
	IND_DTYPE max_output_sz1_max_output_sz1_n1 = max_output_sz1*max_output_sz1*n1;
	
	IND_DTYPE img_sz_img_sz_3 = img_sz*img_sz*3;
	IND_DTYPE img_sz_img_sz = img_sz*img_sz;
	
	for(ind = 0; ind < n_inds; ind++){
		////////////////////////////////////////////// unravel inds
		
		f1 = inds[ind] / (3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = inds[ind] % (3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		channel = r / (s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a1_x = r / (s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a1_y = r / (n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		f2 = r / (s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a2_x = r / (s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a2_y = r / (n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (n3 * s3 * s3 * max_output_sz3 * max_output_sz3);
		
		f3 = r / (s3 * s3 * max_output_sz3 * max_output_sz3);
		r = r % (s3 * s3 * max_output_sz3 * max_output_sz3);
		
		a3_x = r / (s3 * max_output_sz3 * max_output_sz3);
		r = r % (s3 * max_output_sz3 * max_output_sz3);
		
		a3_y = r / (max_output_sz3 * max_output_sz3);
		r = r % (max_output_sz3 * max_output_sz3);
		
		z1 = r / (max_output_sz3);
		z2 = r % (max_output_sz3);
		
		for(img = 0; img < N_IMGS; img++){
			// pool3 -> conv3
			a3_x_global = output_switches3_x[O3_IND(img,f3,z1,z2)] + a3_x;
			a3_y_global = output_switches3_y[O3_IND(img,f3,z1,z2)] + a3_y;
			
			// pool2 -> conv2
			a2_x_global = output_switches2_x[O2_IND(img,f2,a3_x_global,a3_y_global)] + a2_x;
			a2_y_global = output_switches2_y[O2_IND(img,f2,a3_x_global,a3_y_global)] + a2_y;
			
			a1_x_global = output_switches1_x[O1_IND(img,f1,a2_x_global,a2_y_global)] + a1_x;
			a1_y_global = output_switches1_y[O1_IND(img,f1,a2_x_global,a2_y_global)] + a1_y;
			
			if(a1_x_global >= img_sz) /// this shouldn't happen
				a1_x_global = img_sz - 1;
			if(a1_y_global >= img_sz)
				a1_y_global = img_sz - 1;
			
			patches[P_IND(img, ind)] += imgs[I_IND(img, channel,a1_x_global,a1_y_global)];
			channels[P_IND(img, ind)] = channel;
			x[P_IND(img, ind)] = a1_x_global;
			y[P_IND(img, ind)] = a1_y_global;
		} // img
	} // ind
	
	
	list = PyList_New(4);
	if(NULL == list) return NULL;
	if(-1 == PyList_SetItem(list, 0, PyArray_Return(patches_in))) return NULL;
	if(-1 == PyList_SetItem(list, 1, PyArray_Return(channels_in))) return NULL;
	if(-1 == PyList_SetItem(list, 2, PyArray_Return(x_in))) return NULL;
	if(-1 == PyList_SetItem(list, 3, PyArray_Return(y_in))) return NULL;

	return list;
}
