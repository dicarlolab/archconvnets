//N_C * n1 *3 * s1 * s1 * n2 * s2 * s2 * n3 * s3 * s3 * max_output_sz3 * max_output_sz3
#define P_IND(A,B)((B) + (A)*(n_inds))


static PyObject *compute_F_prod_inds(PyObject *self, PyObject *args){
	PyArrayObject *F1_in, *F2_in, *F3_in, *FL_in, *inds_in;
	PyArrayObject *FL321_in;
	
	int dims[14];
	IND_DTYPE *inds;
	float *F1, *F2, *F3, *FL, *FL321;
	
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!",  &PyArray_Type, &F1_in, &PyArray_Type, &F2_in,
		&PyArray_Type, &F3_in, &PyArray_Type, &FL_in, &PyArray_Type, &inds_in)) return NULL;

	if (NULL == F1_in || NULL == F2_in || NULL == F3_in || NULL == FL_in)  return NULL;

	inds = (IND_DTYPE *) inds_in -> data;
	F1 = (float *) F1_in -> data;
	F2 = (float *) F2_in -> data;
	F3 = (float *) F3_in -> data;
	FL = (float *) FL_in -> data;
	
	IND_DTYPE N_C = PyArray_DIM(FL_in, 0);
	IND_DTYPE max_output_sz3 = PyArray_DIM(FL_in, 2);
	IND_DTYPE n3 = PyArray_DIM(F3_in, 0);
	IND_DTYPE n2 = PyArray_DIM(F2_in, 0);
	IND_DTYPE n1 = PyArray_DIM(F1_in, 0);
	IND_DTYPE s1 = PyArray_DIM(F1_in, 2);
	IND_DTYPE s2 = PyArray_DIM(F2_in, 2);
	IND_DTYPE s3 = PyArray_DIM(F3_in, 2);
	IND_DTYPE n_inds = PyArray_DIM(inds_in, 0);
	IND_DTYPE n0 = 3;
	
	dims[0] = N_C;
	dims[1] = n_inds;
	
	FL321_in = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT);
	FL321 = (float *) FL321_in -> data;
	
	int f1, channel, a1_x, a1_y, f2, a2_x, a2_y, f3, a3_x, a3_y, z1, z2, cat, img, ind;
	int a3_x_global, a3_y_global, a2_x_global, a2_y_global, a1_x_global, a1_y_global;
	
	IND_DTYPE r;
	float F321;
	
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
		
		F321 = F1[F1_IND(f1, channel, a1_x, a1_y)] * F2[F2_IND(f2, f1, a2_x, a2_y)] * F3[F3_IND(f3, f2, a3_x, a3_y)];
		for(cat = 0; cat < N_C; cat++){
			FL321[P_IND(cat, ind)] += F321 * FL[FL_IND(cat, f3, z1, z2)];
		} // img
	} // ind
		
	return PyArray_Return(FL321_in);
}
